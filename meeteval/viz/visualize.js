var colormaps = {
    default: {
        'correct': 'lightgray',
        'substitution': '#F5B14D',  // yellow / orange
        'insertion': '#33c2f5', // blue
        'deletion': '#f2beb1',  // red
        // 'ignored': 'transparent',   // purple
        'highlight': 'yellow'
    },
    diff: {
        'correct': 'lightgray',
        'substitution': 'yellow',
        'insertion': 'green',
        'deletion': 'red',
        // 'ignored': 'lightgray',
    },
    seaborn_muted: {
        'correct': 'lightgray',
        'substitution': '#dd8452',  // yellow
        'insertion': '#4c72b0', // blue
        'deletion': '#c44e52',  // red
        // 'ignored': 'lightgray',
    }
}

const constants = {
    utteranceMarkerOverhang: 3,  // Overhang (left and right) of the utterance begin and end markers in pixels
    utteranceMarkerDepth: 6,   // Depth (height) of the utterance marker bracket in pixels
    minStitchOffset: 10,  // Minimum distance of the kink in the stitching line to the word in pixels
    maxViewAreaOverhang: 10, // Maximum time in seconds that the view area can exceed the global domain
    focusZoomThreshold: 45,  // Zoom in if the view area is larger than this threshold when focusing a point / smaller area
    focusMaxViewAreasize: 30,  // Maximum view area size when focusing on a point / small area
};

/**
 * Throttles a function call. Tracks different instances via `object`.
 *
 * Example call: `call_throttled(() => console.log('call')), trackerObject);`
 * Distance between calls is at least `delay` ms.
 * If the code is called again within the delay, the call is postponed.
 */
function call_throttled(fn, object, delay=10) {
    // Expects that the function does always the same thing.
    // The object is used to store the timerId and the call_pending flag.
    // (Cannot be stored in fn, because it is created with bind.)
    object.timerId_fn = fn
    if (!object.timerId) {
        // Call immediately
        fn()

        // Set timer to prevent further calls
        object.timerId = setTimeout(() => {
            object.timerId = undefined;
            if (object.call_pending) object.timerId_fn();
            object.call_pending = false;
        }, delay);
    } else {object.call_pending = true;}
}

/**
 * Call `fn` earliest after `delay` ms, but only if no other call is made in the meantime.
 * If the function is called again within `delay` ms, the first call is canceled
 * and the second call is postponed by `delay` ms.
 *
 * Example call: `call_delayed_throttled(this.setURL.bind(this), this.setURL, 200);`
 */
function call_delayed_throttled(fn, object, delay=5) {
    if (object.timerId) {
        clearTimeout(object.timerId);
    }
    object.timerId = setTimeout(() => {
        fn()
        object.timerId = undefined;
    }, delay);
}

/**
 * Interpolate between a and b with ratio c/d.
 * If c <= 0, return a, if c >= d, then return b.
 */
function interpolate (a, b, c, d) {
    if (c <= 0 || !b) {
        // For speedup and numeric
        return a
    } else if (c >= d) {
        // For speedup and numeric
        return b
    } else {
        // Interpolate such that the length ratio between c and c+1 is constant.
        const length_a = a[1] - a[0];
        const length_b = b[1] - b[0];
        const log_length_a = Math.log(length_a);
        const log_length_b = Math.log(length_b);

        const length = Math.exp(log_length_b  + ((log_length_a - log_length_b) * (d-c) / d));

        let ratio = (length - length_b) / (length_a - length_b)

        if (isNaN(ratio)) return a;

        ratio = 1 - Math.max(Math.min(ratio, 1), 0);
        let ratio_2 = 1 - ratio
        return [a[0] * ratio_2 + b[0] * ratio, a[1] * ratio_2 + b[1] * ratio]
    }
}

/**
 * Checks whether two ranges `a` and `b` are similar by relative tolerance `tolerance`.
 */
function similar_range(a, b, tolerance=0.00001){
    // Note: a == [0, 1] will always return false, because arrays are compared by reference, not values
    if (JSON.stringify(a) === JSON.stringify(b)) {
        return true
    }

    if (!a || !b || a.some(isNaN) || b.some(isNaN))
        return false

    const delta = Math.min(a[1] - a[0], b[1] - b[0]) * tolerance

    if (a.length !== b.length) {
        return false;
    }
    for (var i = 0; i < a.length; i++) {
        if (Math.abs(a[i] - b[i]) > delta) {
            return false; // Return false if any elements are different
        }
    }
    return true
}


function brushExceedsViewArea(viewArea, brushArea, tolerance=0.00001) {
    const delta = Math.min(brushArea[1] - brushArea[0], viewArea[1] - viewArea[0]) * tolerance
    return brushArea[0] < viewArea[0] + delta && brushArea[1] > viewArea[1] - delta;
}

/**
 * Moves the view area `v` into `bound` if at least one boundary of `v` lies outside of `bound`.
 * If the size of `v` is larger than `bound`, `v` is clipped to `bound`, resulting in `v = bound`.
 */
function moveOrClip(v, bound) {
    let dirty = false;
    if (v[0] < bound[0]) {
        v = [bound[0], v[1] + bound[0] - v[0]];
        dirty = true;
    } else if (v[1] > bound[1]) {
        v = [v[0] + bound[1] - v[1], bound[1]];
        dirty = true;
    }
    if (v[0] < bound[0] || v[1] > bound[1]) {
        // This can only happen if one of the previous if statements is true, 
        // so dirty is alread set
        v = bound;
    }
    return [v, dirty];
}


/**
 * Ensures that every entry (index i) lies within its parent (index i - 1).
 *
 * Keeps the viewArea at `anchor` fixed if it doesn't exceed viewAreas[0].
 *
 * Returns an array of booleans indicating which viewAreas were changed.
 */
function adjustViewAreas(viewAreas, anchor, viewArea) {
    const dirty = viewAreas.map(() => false);

    const oldViewAreas = [...viewAreas];

    // Limit the viewArea to the global domain +/- constants.maxViewAreaOverhang seconds
    const maxArea = [viewAreas[0][0] - constants.maxViewAreaOverhang, viewAreas[0][1] + constants.maxViewAreaOverhang];

    // Limit zoom to 0.1 seconds
    const minAreaLength = 0.1;


    /**
     * Moves or extends the view area `v` such that `bound` lies within `v`.
     */
    function moveOrExtend(v, bound) {
        let dirty = false;
        if (bound[0] < v[0]) {
            v = [bound[0], v[1] + bound[0] - v[0]];
            dirty = true;
        } else if (bound[1] > v[1]) {
            v = [v[0] + bound[1] - v[1], bound[1]];
            dirty = true;
        }
        if (bound[0] < v[0] || bound[1] > v[1]) {
            // This can only happen if one of the previous if statements is true, 
            // so dirty is alread set
            v = bound;
        }
        return [v, dirty];
    }

    // Set the view Area at anchor. Only allow viewAreas that are larger than
    // the minimum width
    const center = (viewArea[0] + viewArea[1]) / 2;
    let d;
    [viewArea, d] = moveOrExtend(viewArea, [center - minAreaLength / 2, center + minAreaLength / 2])
    if (d) console.log('Maximum zoom level reached');
    dirty[anchor] = !similar_range(viewAreas[anchor], viewArea);
    [viewAreas[anchor], d] = moveOrClip(viewArea, maxArea);
    if (d) console.log('Maximum view area reached');

    // Update children of anchor (j > i)
    // View areas are moved or clipped so that they lie within their parent.
    // No check for the max viewArea is necessary here, because viewAreas[anchor]
    // already fulfills this condition.
    for (let j = anchor + 1; j < viewAreas.length; j++) {
        if (similar_range(oldViewAreas[j - 1], viewAreas[j])) {
            // Keep the view area of the parent if no brush is visible (i.e., viewAreas[j] == viewAreas[j-1])
            dirty[j] = !similar_range(viewAreas[j], viewAreas[j - 1]);
            viewAreas[j] = viewAreas[j - 1];
        } else {
            [viewAreas[j], dirty[j]] = moveOrClip(viewAreas[j], viewAreas[j - 1]);
        }
    }

    // Update parents of anchor (j < i). Index 0 is the global domain and
    // should not be changed.
    // We don't have to clip to the maxArea because the anchor is already 
    // clipped to the maxArea and no other viewArea can be larger (by invariant).
    for (let j = anchor - 1; j > 0; j--) {
        [viewAreas[j], dirty[j]] = moveOrExtend(viewAreas[j], viewAreas[j + 1]);
    }

    return dirty;
}


//<!--!Font Awesome Free 6.5.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.-->
// Add new icons:
//  - Open e.g. https://fontawesome.com/icons/copy?f=classic&s=solid
//  - Click SVG
//  - Copy source code and append it to the following object (key is taken from HTML code, e.g. <i class="fa-solid fa-copy"></i>)
//  - Add `class="icon" aria-hidden="true" focusable="false"` to the source code.
var _icons = {
    'fa-solid fa-sliders': '<svg xmlns="http://www.w3.org/2000/svg" class="icon" aria-hidden="true" focusable="false" viewBox="0 0 512 512"><path d="M0 416c0 17.7 14.3 32 32 32l54.7 0c12.3 28.3 40.5 48 73.3 48s61-19.7 73.3-48L480 448c17.7 0 32-14.3 32-32s-14.3-32-32-32l-246.7 0c-12.3-28.3-40.5-48-73.3-48s-61 19.7-73.3 48L32 384c-17.7 0-32 14.3-32 32zm128 0a32 32 0 1 1 64 0 32 32 0 1 1 -64 0zM320 256a32 32 0 1 1 64 0 32 32 0 1 1 -64 0zm32-80c-32.8 0-61 19.7-73.3 48L32 224c-17.7 0-32 14.3-32 32s14.3 32 32 32l246.7 0c12.3 28.3 40.5 48 73.3 48s61-19.7 73.3-48l54.7 0c17.7 0 32-14.3 32-32s-14.3-32-32-32l-54.7 0c-12.3-28.3-40.5-48-73.3-48zM192 128a32 32 0 1 1 0-64 32 32 0 1 1 0 64zm73.3-64C253 35.7 224.8 16 192 16s-61 19.7-73.3 48L32 64C14.3 64 0 78.3 0 96s14.3 32 32 32l86.7 0c12.3 28.3 40.5 48 73.3 48s61-19.7 73.3-48L480 128c17.7 0 32-14.3 32-32s-14.3-32-32-32L265.3 64z"/></svg>',
    'fa-circle-question': '<svg xmlns="http://www.w3.org/2000/svg" class="icon" aria-hidden="true" focusable="false" viewBox="0 0 512 512"><path d="M464 256A208 208 0 1 0 48 256a208 208 0 1 0 416 0zM0 256a256 256 0 1 1 512 0A256 256 0 1 1 0 256zm169.8-90.7c7.9-22.3 29.1-37.3 52.8-37.3h58.3c34.9 0 63.1 28.3 63.1 63.1c0 22.6-12.1 43.5-31.7 54.8L280 264.4c-.2 13-10.9 23.6-24 23.6c-13.3 0-24-10.7-24-24V250.5c0-8.6 4.6-16.5 12.1-20.8l44.3-25.4c4.7-2.7 7.6-7.7 7.6-13.1c0-8.4-6.8-15.1-15.1-15.1H222.6c-3.4 0-6.4 2.1-7.5 5.3l-.4 1.2c-4.4 12.5-18.2 19-30.6 14.6s-19-18.2-14.6-30.6l.4-1.2zM224 352a32 32 0 1 1 64 0 32 32 0 1 1 -64 0z"/></svg>',
    'fa-solid fa-volume-high': '<svg xmlns="http://www.w3.org/2000/svg" class="icon" aria-hidden="true" focusable="false" viewBox="0 0 640 512"><path d="M533.6 32.5C598.5 85.2 640 165.8 640 256s-41.5 170.7-106.4 223.5c-10.3 8.4-25.4 6.8-33.8-3.5s-6.8-25.4 3.5-33.8C557.5 398.2 592 331.2 592 256s-34.5-142.2-88.7-186.3c-10.3-8.4-11.8-23.5-3.5-33.8s23.5-11.8 33.8-3.5zM473.1 107c43.2 35.2 70.9 88.9 70.9 149s-27.7 113.8-70.9 149c-10.3 8.4-25.4 6.8-33.8-3.5s-6.8-25.4 3.5-33.8C475.3 341.3 496 301.1 496 256s-20.7-85.3-53.2-111.8c-10.3-8.4-11.8-23.5-3.5-33.8s23.5-11.8 33.8-3.5zm-60.5 74.5C434.1 199.1 448 225.9 448 256s-13.9 56.9-35.4 74.5c-10.3 8.4-25.4 6.8-33.8-3.5s-6.8-25.4 3.5-33.8C393.1 284.4 400 271 400 256s-6.9-28.4-17.7-37.3c-10.3-8.4-11.8-23.5-3.5-33.8s23.5-11.8 33.8-3.5zM301.1 34.8C312.6 40 320 51.4 320 64V448c0 12.6-7.4 24-18.9 29.2s-25 3.1-34.4-5.3L131.8 352H64c-35.3 0-64-28.7-64-64V224c0-35.3 28.7-64 64-64h67.8L266.7 40.1c9.4-8.4 22.9-10.4 34.4-5.3z"/></svg>',
    'fa-solid fa-check': '<svg xmlns="http://www.w3.org/2000/svg" class="icon" aria-hidden="true" focusable="false" viewBox="0 0 448 512"><path d="M438.6 105.4c12.5 12.5 12.5 32.8 0 45.3l-256 256c-12.5 12.5-32.8 12.5-45.3 0l-128-128c-12.5-12.5-12.5-32.8 0-45.3s32.8-12.5 45.3 0L160 338.7 393.4 105.4c12.5-12.5 32.8-12.5 45.3 0z"/></svg>',
    'fa-solid fa-copy': '<svg xmlns="http://www.w3.org/2000/svg" class="icon" aria-hidden="true" focusable="false" viewBox="0 0 448 512"><path d="M208 0H332.1c12.7 0 24.9 5.1 33.9 14.1l67.9 67.9c9 9 14.1 21.2 14.1 33.9V336c0 26.5-21.5 48-48 48H208c-26.5 0-48-21.5-48-48V48c0-26.5 21.5-48 48-48zM48 128h80v64H64V448H256V416h64v48c0 26.5-21.5 48-48 48H48c-26.5 0-48-21.5-48-48V176c0-26.5 21.5-48 48-48z"/></svg>',
    'fa-solid fa-caret-up': '<svg xmlns="http://www.w3.org/2000/svg" class="icon" aria-hidden="true" focusable="false" viewBox="0 0 320 512"><path d="M182.6 137.4c-12.5-12.5-32.8-12.5-45.3 0l-128 128c-9.2 9.2-11.9 22.9-6.9 34.9s16.6 19.8 29.6 19.8H288c12.9 0 24.6-7.8 29.6-19.8s2.2-25.7-6.9-34.9l-128-128z"/></svg>',
    'fa-solid fa-caret-down': '<svg xmlns="http://www.w3.org/2000/svg" class="icon" aria-hidden="true" viewBox="0 0 320 512"><path d="M137.4 374.6c12.5 12.5 32.8 12.5 45.3 0l128-128c9.2-9.2 11.9-22.9 6.9-34.9s-16.6-19.8-29.6-19.8L32 192c-12.9 0-24.6 7.8-29.6 19.8s-2.2 25.7 6.9 34.9l128 128z"/></svg>',
    'fa-solid fa-triangle-exclamation': '<svg xmlns="http://www.w3.org/2000/svg" class="icon" aria-hidden="true" focusable="false" viewBox="0 0 512 512"><path d="M256 32c14.2 0 27.3 7.5 34.5 19.8l216 368c7.3 12.4 7.3 27.7 .2 40.1S486.3 480 472 480H40c-14.3 0-27.6-7.7-34.7-20.1s-7-27.8 .2-40.1l216-368C228.7 39.5 241.8 32 256 32zm0 128c-13.3 0-24 10.7-24 24V296c0 13.3 10.7 24 24 24s24-10.7 24-24V184c0-13.3-10.7-24-24-24zm32 224a32 32 0 1 0 -64 0 32 32 0 1 0 64 0z"/></svg>',
}
var icons = {
    'menu': _icons['fa-solid fa-sliders'],
    'audio': _icons['fa-solid fa-volume-high'],
    'help': _icons['fa-circle-question'],
    'warning': _icons['fa-solid fa-triangle-exclamation'],
    'check': _icons['fa-solid fa-check'],
    'copy': _icons['fa-solid fa-copy'],
    'caret-up': _icons['fa-solid fa-caret-up'],
    'caret-down': _icons['fa-solid fa-caret-down'],
}

function alignment_visualization(
    data,
    element_id = '#my_dataviz',
    settings= {
        colors: colormaps.default,
        barplot: {
            style: 'relative', // hidden, absolute, relative
            scaleExcludeCorrect: false
        },
        minimaps: {
            number: 2,
        },
        font_size: 12,
        search_bar: {
            initial_query: null
        },
        recording_file: "",
        match_width: 10,  // width between ref and hyp boxes, i.e., the space for the alignment lines
        audio_server: 'http://localhost:7777',
        syncID: null,
        encodeURL: true,
        show_playhead: true,    // Show the current position of audio playback in the details plot
    }
) {

    window.parent.postMessage({
        type: 'url',
        url: window.location.href,
        }, '*'
    )

    if (settings.font_size === undefined) {
        // The default from the function signature doesn't work.
        settings.font_size = 12;
    }

    var urlParams = new URLSearchParams(window.location.search);
    if (settings.encodeURL && urlParams.has('minimaps')) {
        settings.minimaps.number = urlParams.get('minimaps')
    }
    if (settings.encodeURL && urlParams.has('regex')) {
        settings.search_bar.initial_query = urlParams.get('regex');
    }

    // Decompress data
    var transposed_words = []
    Object.keys(data.words).forEach(function(key) {
        data.words[key].forEach(function(value, index) {
            if (!transposed_words[index]) {
                transposed_words[index] = {};
            }
            transposed_words[index][key] = value;
        });
    });
    transposed_words.forEach(function(word, index) {
        word.end_time = word.start_time + word.duration;
        word.center_time = (word.start_time + word.end_time) / 2;  // Point where the stitches attach
        if (word.source == 'h') {word.source = 'hypothesis';}
        else if (word.source == 'r') {word.source = 'reference';}
        word.matches?.forEach(function(match) {
            switch (match[1]) {
                case 'c': match[1] = 'correct'; break;
                case 's': match[1] = 'substitution'; break;
                case 'd': match[1] = 'deletion'; break;
                case 'i': match[1] = 'insertion'; break;
            }
        })
    });
    data.words = transposed_words;

    // Set the custom locale globally
    d3.formatDefaultLocale({
        "decimal": ".",
        "thousands": "\u202F",  // https://en.wikipedia.org/wiki/Decimal_separator#Digit_grouping
        "grouping": [3], // Specify the grouping of thousands (e.g., [3] means every 3 digits)
        // "currency": ["$", ""], // Optional: specify currency symbols
        // "dateTime": "%a %b %e %X %Y", // Optional: specify date/time formats
    });

    // Validate settings
    for (const label of ['correct', 'substitution', 'insertion', 'deletion']) {
        if (settings.colors[label] === undefined) throw `Missing key in "colors" setting: ${label}`;
    }

    function parseSelection(selection) {
        const match = /^(?<start>[+-]?([0-9]*[.])?[0-9]+)\s*-\s*(?<end>[+-]?([0-9]*[.])?[0-9]+)$/.exec(selection);

        if (match) {
            const start = parseFloat(match.groups.start);
            const end = parseFloat(match.groups.end);

            // Check for invalid ranges. Return null if invalid.
            if (isNaN(start) || isNaN(end) || start > end) return null;

            return [start, end];
        }
    }

    // Data preprocessing. Determine displayed domain from the data and add a margin of 1s
    const time_domain = [
        Math.min.apply(null, (data.utterances.map(d => d.start_time))) - 1, 
        Math.max.apply(null, (data.utterances.map(d => d.end_time))) + 1
    ];

    // Get speakers and sort for side-by-side view. The side-by-side view is confusing 
    // when the plots have different speaker orders
    const speakers = [...new Set(data.utterances.map(d => d.speaker))];
    speakers.sort();

    // Global state management
    // Constraint: every view area (i) must be contained within its parent (i-1)
    let state = {
        // State of minimaps and details plot
        viewAreas: [],
        filteredWords: [],
        //
        selectedSegment: null,
        // This scale is shared between all plots
        speakerScale: d3.scaleBand().domain(speakers).padding(0.1),
        words: data.words,
        // Track which viewAreas changed
        dirty: [],
        // Track current playback time
        playbackTime: null,
        playbackDirty: false,
    };

    function initializeViewAreas(domain, finalViewArea=null) {
        state.viewAreas = [];
        var urlParams = new URLSearchParams(window.location.search);
        if (finalViewArea === null) {
            if (settings.encodeURL && urlParams.has('selection')) {
                console.log("Setting selection from URL", urlParams.get('selection'));
                finalViewArea = parseSelection(urlParams.get('selection'));
                if (!finalViewArea) {
                    console.log("Invalid selection in URL. Using full domain.");
                    finalViewArea = domain;
                }
            } else {
                finalViewArea = domain;
            }
        }

        for (let i = 0; i < settings.minimaps.number; i++) {
            const viewArea = interpolate([...domain], [...finalViewArea], i, settings.minimaps.number);
            state.viewAreas.push(viewArea);
        }
        state.viewAreas.push([...finalViewArea]);
        state.dirty = state.viewAreas.map(() => true);

        // Reset filtered words after initialization
        state.filteredWords = [];
    }

    initializeViewAreas(time_domain);

    function setViewArea(i, viewArea) {
        const dirty = adjustViewAreas(state.viewAreas, i, viewArea);
        for (let j = 0; j < state.viewAreas.length; j++) {
            state.dirty[j] |= dirty[j];
        }
    }

    // Tracker objects for throttling
    const urlTracker = {};
    const drawTracker = {};
    /**
     * Updates the views to be consistent with the current state.
     * Includes the plots, the selection and the URL.
     */
    function update() {
        call_throttled(() => {
            const dirty = [...state.dirty];
            state.dirty.fill(false);

            // Update minimaps
            minimaps.forEach((minimap, j) => {
                const viewArea = state.viewAreas[j];
                if (dirty[j]) {
                    state.filteredWords[j] = (state.filteredWords[j - 1] ?? data.words)
                        .filter(w => w.start_time < viewArea[1] && w.end_time > viewArea[0])
                }
                if (dirty[j])
                    minimap.updateViewArea();
                if (dirty[j] || dirty[j + 1])
                    minimap.updateBrush()
            });

            // Update details plot
            if (dirty[dirty.length - 1]) {
                const viewArea = state.viewAreas[state.viewAreas.length - 1];
                state.filteredWords[dirty.length - 1] = (state.filteredWords[dirty.length - 2] ?? data.words)
                    .filter(w => w.start_time < viewArea[1] && w.end_time > viewArea[0])
                details_plot.update(
                    viewArea,
                    state.filteredWords[state.filteredWords.length - 1],
                );
            }

            // Update playhead
            if (state.playbackDirty) {
                details_plot.updatePlayhead(state.playbackTime);
                state.playbackDirty = false;
            }
        }, drawTracker, 20);

         // Update URL
        if (settings.encodeURL) call_delayed_throttled(
            () => {
                const selection = state.viewAreas[state.viewAreas.length - 1];
                set_url_param('selection', `${selection[0].toFixed(1)}-${selection[1].toFixed(1)}`)
            },
            urlTracker,
            200
        )

        // Update range selector field. This is cheap, so no throttling
        rangeSelector.update(state.viewAreas[state.viewAreas.length - 1]);

        // Send message to window for synchronization. We want this to be
        // instantanous, so no throttling.
        if (!handlingWindowEvent) {
            window.parent.postMessage({
                type: 'viewAreas',
                viewAreas: state.viewAreas,
                syncID: settings.syncID,
                sourceID: element_id,
                }, '*'
            )
        }
    }

    let handlingWindowEvent = false;
    function updateViewArea(i, viewArea) {
        if (similar_range(state.viewAreas[i], viewArea)) return;
        setViewArea(i, viewArea);
        update();
    }

    /**
     * Focus the view on a point.
     * 
     * Animates the details plot and all minimaps such that the point is visible in the details view.
     * Centers the point if it was not visible before or if moveIfInViewport=true.
     * 
     * Zooms in such that 30s are visible if the current zoom level is larger than 45s. This is to make sure
     * that the text is readable.
     */
    function focusPoint(point, moveIfInViewport=false) {
        const [start, end] = state.viewAreas[state.viewAreas.length - 1];
        let currentViewAreaSize = end - start;

        // Nothing to do if the point is already in the viewport and not zoomed out too much
        if (!moveIfInViewport && point > start && point < end && (currentViewAreaSize < constants.focusZoomThreshold)) return;

        // Limit the size of the viewport to maxViewAreaSize if the threshold is exceeded
        if (currentViewAreaSize > constants.focusZoomThreshold) {
            currentViewAreaSize = constants.focusMaxViewAreasize;
        }

        const viewArea = [point - currentViewAreaSize / 2, point + currentViewAreaSize / 2];

        animateToViewArea(state.viewAreas.length - 1, viewArea, moveIfInViewport);
    }

    /**
     * Focus the view on an area.
     * 
     * Animates the details plot and all minimaps such that the area is in the center of the details view.
     * 
     * Zooms out if the area is larger than the current view area. Zooms in if the area is smaller than the 
     * current view area and the current view area is larger than 45s. This is to prevent exessively changing 
     * the zoom level.
     */
    function focusArea(area, moveIfInViewport=true, margin=1) {
        const [start, end] = state.viewAreas[state.viewAreas.length - 1];
        let viewAreaCenter = (start + end) / 2;
        let viewAreaSize = end - start;
        const areaSize = area[1] - area[0] + margin;
        const areaCenter = (area[0] + area[1]) / 2;

        // Zoom in if the current view area is too large
        if (areaSize < constants.focusZoomThreshold && viewAreaSize > constants.focusZoomThreshold) {
            viewAreaSize = Math.max(constants.focusMaxViewAreasize, areaSize);
        }

        // Zoom out if the area is larger than the current view area
        if (viewAreaSize < areaSize) {
            viewAreaSize = areaSize;
        }

        // Center to the area if moving is requested or the area is not in the viewport
        if (moveIfInViewport || areaCenter - areaSize / 2 < start || areaCenter + areaSize / 2 > end) {
            viewAreaCenter = areaCenter;
        }

        animateToViewArea(state.viewAreas.length - 1, [viewAreaCenter - viewAreaSize / 2, viewAreaCenter + viewAreaSize / 2]);
    }

    let animationIntervalID = null;
    function animateToViewArea(i, viewArea) {
        console.log('Animate to view area', i, viewArea);
        if (viewArea[0] > viewArea[1]) {
            console.error('Invalid view area', viewArea);
            return;
        }

        // Clip to the max allowed view area by the global domain. This is not strictly required 
        // since it also happens in setViewArea, but makes the animation smoother if the animation
        // ends at the correct position.
        const maxArea = [state.viewAreas[0][0] - constants.maxViewAreaOverhang, state.viewAreas[0][1] + constants.maxViewAreaOverhang];
        const target_location = moveOrClip(viewArea, maxArea)[0];

        // Animate view area to the location
        clearInterval(animationIntervalID);

        // Animate top and bottom independently exponentially
        const step = () => {
            const currentLocation = state.viewAreas[state.viewAreas.length - 1];
            const distance0 = currentLocation[0] - target_location[0];
            const distance1 = currentLocation[1] - target_location[1];
            let step0 = Math.min(Math.abs(distance0) / 4 + 0.05, Math.abs(distance0));
            let step1 = Math.min(Math.abs(distance1) / 4 + 0.05, Math.abs(distance1));
            // Limit the speed of the faster moving point such that both end at the same time
            if (step0 > step1) {
                if (distance0 > 0) step1 = step0 * Math.abs(distance1 / distance0);
            } else {
                if (distance1 > 0) step0 = step1 * Math.abs(distance0 / distance1);
            }
            step0 *= Math.sign(distance0);
            step1 *= Math.sign(distance1);
            setViewArea(i, [currentLocation[0] - step0, currentLocation[1] - step1]); 
            update();
            if (distance0 == 0 && distance1 == 0) clearInterval(animationIntervalID);
        };

        // 20ms is the throttling interval for update()
        animationIntervalID = setInterval(step, 25);
        step(); // Do first update immediately for instant feedback
    }

    /**
     * Selects a segment and updates the details plot.
     *  
     * If `focus` is true, the view area is moved such that the segment is in the center of the view area.
     * If `keepZoom` is true, the zoom level is kept. If `keepZoom` is false, the zoom level is adjusted to the segment.
     * If `keepZoom` is auto, the zoom level is increased until no more than 15s are visible. If the segment is larger 
     * than 15s, the zoom level is adjusted to the segment.
     */
    function selectSegment(segment, focus=false) {
        if (state.selectedSegment != segment) {
            state.selectedSegment = segment;
            state.dirty[state.dirty.length - 1] = true;
            selectedUtteranceDetails.update(segment)
            update();
        }

        if (focus && segment) {
            focusArea([segment.start_time, segment.end_time]);
        }
    }

    /**
     * Selects the next segment for which condition(segment) is true.
     * 
     * If no segment is selected, the search begins at the beginning.
     * 
     * If no next segment is found for which condition is true, the segment will be unselected.
     */
    function selectNextMatchingSegment(condition, focus=true, reverse=false) {
        let candidates;
        if (reverse) {
            candidates = data.utterances.slice(0, state.selectedSegment?.utterance_index).reverse();
        } else {
            candidates = data.utterances.slice(state.selectedSegment?.utterance_index + 1);
        }
        const segment = candidates.find(condition);
        selectSegment(segment, focus);
        return segment;
    }

    if (settings.syncID !== null) {
        window.addEventListener("message", event => {
            if (
                event.data.type === 'viewAreas' &&
                event.data.syncID === settings.syncID &&
                event.data.sourceID !== element_id
            ) {
                handlingWindowEvent = true;
                state.viewAreas.forEach((viewArea, i) => {
                    if (i === 0) return;
                    const newViewArea = event.data.viewAreas[i];
                    if (i === state.viewAreas.length - 1)
                        updateViewArea(state.viewAreas.length - 1, event.data.viewAreas[event.data.viewAreas.length - 1]);
                    else if (newViewArea)
                        updateViewArea(i, newViewArea);
                    else {
                        updateViewArea(i, event.data.viewAreas[event.data.viewAreas - 1]);
                    }
                });
                handlingWindowEvent = false;
            }
        })
    }

    let root_element = d3.select(element_id);

    /* Mouse drag */
    let dragActive = false; // We can only have one drag at a time globally

    /**
     * Start dragging an element. Do nothing if a drag is currently active.
     *
     * Registers global handlers for mousemove and mouseup events
     * so that the mouse can be dragged out of the element or window
     * and still be captured. Cancel defaults so that text is not
     * selected during dragging.
     */
    function drag(element, drag, startDrag, stopDrag) {
        function _stopDrag(e) {
                window.removeEventListener("mousemove", _drag);
                window.removeEventListener("mouseup", _stopDrag);
                dragActive = false;
                if (stopDrag) stopDrag(e);
                this.lastY = undefined;
            }


        function _drag(e) {
            if (!this.lastY) {
                this.lastY = e.y
                drag(e)
            } else {
                // e.movementY: CB: Movement for canvas? It is too large for the details plot (maybe 20%).
                drag(e, delta_y=e.y - this.lastY);
                this.lastY = e.y;
            };
            e.preventDefault();
        }

        function _startDrag() {
            if (dragActive) return;
            dragActive = true;
            window.addEventListener("mousemove", _drag);
            window.addEventListener("mouseup", _stopDrag);
            if (startDrag) startDrag();
        }
        element.on("mousedown", _startDrag);
    }


class Axis {
    // all y axes
    constructor(padding, numTicks=null, tickPadding=3, tickSize=6) {
        this._padding = padding;
        this._tickPadding = tickPadding;
        this._tickSize = tickSize;
        this.horizontal = true;
        this.numTicks = numTicks;
        this._update_dpr();
    }

    _update_dpr() {
        this.dpr = window.devicePixelRatio || 1;
        this.padding = this._padding * this.dpr;
        this.tickPadding = this._tickPadding * this.dpr;
        this.tickSize = this._tickSize * this.dpr;
    }

    draw(
        context,
        scale,
        position,
    ) {
        this._update_dpr();

        const [start, end] = scale.range(),
            tickFormat = scale.tickFormat ? scale.tickFormat() : d => d,
            ticks = (scale.ticks ? scale.ticks(this.numTicks) : scale.domain()).map(d => {
                return {
                    pos: scale(d) + (scale.bandwidth ? scale.bandwidth() / 2 : 0),
                    label: tickFormat(d)
                }
            });

        // Flip coords if vertical
        let coord, c;
        if (this.horizontal) {
            coord = (x, y) => [x, position + y]
            c = (x, y) => [x, y]
        } else {
            coord = (x, y) => [position - y, x]
            c = (x, y) => [-y, x]
        }

        // Set up context
        context.lineWidth = 1;
        context.font = `${12 * this.dpr}px Arial`;
        context.strokeStyle = "black";  // Line color
        context.fillStyle = "black";    // Font color

        // Clear the axis part of the plot
        context.clearRect(...coord(start, 0), ...c(end - start, this.padding));

        // Tick marks
        context.beginPath();
        ticks.forEach(d => {
            context.moveTo(...coord(d.pos, 0));
            context.lineTo(...coord(d.pos, this.tickSize));
        });
        // Line
        context.moveTo(...coord(start, this.tickSize));
        context.lineTo(...coord(start, 0));
        context.lineTo(...coord(end, 0));
        context.lineTo(...coord(end, this.tickSize));
        context.stroke();

        // Tick labels
        if (this.horizontal) {
            context.textAlign = "center";
            context.textBaseline = "top";
        } else {
            context.textAlign = "right";
            context.textBaseline = "middle";
        }

        ticks.forEach(d => {
            context.fillText(d.label, ...coord(d.pos, this.tickSize + this.tickPadding));
        });
    }
}

class CompactAxis {
    // x axis of minimap
    constructor(padding, label=null) {
        this._padding = padding;
        this.label = label;
        this._update_dpr();
    }

    _update_dpr() {
        this.dpr = window.devicePixelRatio || 1;
        this.padding = this._padding * this.dpr;
    }

    draw(
        context,
        scale,
        position,
    ) {
        this._update_dpr();

        // Tick labels
        context.textAlign = "center";
        context.textBaseline = "top";

        const [start, end] = scale.range();
        const tickFormat = scale.tickFormat ? scale.tickFormat() : d => d;
        const ticks = (scale.ticks ? scale.ticks() : scale.domain()).map(d => {
            const label = tickFormat(d);
            return {
                pos: scale(d) + (scale.bandwidth ? scale.bandwidth() / 2 : 0),
                label: label,
                textMetrics: context.measureText(label)
            }
        });

        // Flip coords if vertical
        let coord, c;
        coord = (x, y) => [x, position + y]
        c = (x, y) => [x, y]

        // Set up context
        context.lineWidth = 1;
        context.font = `${12 * this.dpr}px Arial`;
        context.strokeStyle = "black";  // Line color
        context.fillStyle = "black";    // Font color

        // Clear the axis part of the plot
        context.clearRect(...coord(start, 0), ...c(end - start, this.padding));

        context.beginPath();
        // Line
        const p = (ticks[0].textMetrics.fontBoundingBoxAscent + ticks[0].textMetrics.fontBoundingBoxDescent) / 2;
        context.moveTo(...coord(start, p));
        context.lineTo(...coord(end, p));
        context.stroke();

        ticks.forEach(d => {
            context.clearRect(
                ...coord(d.pos - d.textMetrics.width / 2 - 2, -d.textMetrics.fontBoundingBoxAscent),
                d.textMetrics.width + 4,
                d.textMetrics.fontBoundingBoxDescent + d.textMetrics.fontBoundingBoxAscent
            );
            context.fillText(d.label, ...coord(d.pos, 0));
        });

        if (this.label) {
            context.textAlign = "right";
            const textMetrics = context.measureText(this.label);
            const x = scale.range()[1];
            context.clearRect(x - textMetrics.width - 3, position - textMetrics.fontBoundingBoxAscent, textMetrics.width + 3, textMetrics.fontBoundingBoxAscent + textMetrics.fontBoundingBoxDescent);
            context.fillText(this.label, x, position);
        }
    }
}

class DetailsAxis{
    // x axis of details plot (the plot at the bottom)
    constructor(padding, tickPadding=3, tickSize=6) {
        this._padding = padding;
        this._tickPadding = tickPadding;
        this._tickSize = tickSize;
        this._update_dpr();
    }

    _update_dpr() {
        this.dpr = window.devicePixelRatio || 1;
        this.padding = this._padding * this.dpr;
        this.tickPadding = this._tickPadding * this.dpr;
        this.tickSize = this._tickSize * this.dpr;
    }

    draw(context, scale, position) {
        this._update_dpr();

        const [start, end] = scale.range()
        const ticks = scale.domain().map(d => {
            return {
                pos: scale(d) + (scale.bandwidth ? scale.bandwidth() / 2 : 0),
                label: d
            }
        })
        const offset = (scale.bandwidth()) / 4;
        const match_width = settings.match_width * scale.bandwidth() / 2;

        // Clear the axis part of the plot
        context.clearRect(start, position, end - start, this.padding);

        // Set up context
        context.lineWidth = 1;
        context.font = `${12 * this.dpr}px Arial`;
        context.strokeStyle = "black";  // Line color
        context.fillStyle = "black";    // Font color

        // Line
        context.moveTo(start, position);
        context.lineTo(end, position);
        context.stroke();

        context.textAlign = "center";
        context.textBaseline = "top";
        ticks.forEach(d => {
            context.beginPath();
            context.rect(d.pos - scale.bandwidth() / 2, position, scale.bandwidth(), this.padding);
            context.fillStyle = "white";
            context.strokeStyle = "black";
            context.fill();
            context.stroke();
            context.beginPath();
            context.moveTo(d.pos - match_width, position);
            context.lineTo(d.pos - match_width, position + this.tickSize);
            context.moveTo(d.pos + match_width, position);
            context.lineTo(d.pos + match_width, position + this.tickSize);
            context.moveTo(d.pos + scale.bandwidth() / 2, position);
            context.lineTo(d.pos + scale.bandwidth() / 2, position + this.tickSize);
            context.moveTo(d.pos - scale.bandwidth() / 2, position);
            context.lineTo(d.pos - scale.bandwidth() / 2, position + this.tickSize);
            context.stroke();
            context.fillStyle = "black";
            context.fillText(d.label, d.pos, position + this.tickSize + 2 * this.tickPadding)
            context.fillText("REF", d.pos - offset, position + this.tickPadding);
            context.fillText("HYP", d.pos + offset, position + this.tickPadding);
        });
    }
}

/**
 * Add a tooltip to an element. The tooltip is shown when the mouse
 * enters the element and hidden when the mouse leaves the element.
 * The tooltip is positioned below the element and shifted so that it
 * is fully visible and, if possible, centered below the element.
 *
 * @param {d3.Selection} element The element to add the tooltip to
 * @param {string|function} tooltip The content of the tooltip. If a
 * function is provided, it is called with the tooltip element as
 * argument.
 * @param {function} preShow A function that is called before the
 * tooltip is shown. This can be used to update the tooltip content
 * dynamically. The tooltip position is corrected after this function
 * is called.
 */
function addTooltip(element, tooltip, preShow) {
    element.classed("tooltip", true);
    const tooltipcontent = element.append("div").classed("tooltipcontent", true);
    if (typeof tooltip === "string") tooltipcontent.text(tooltip)
    else if (tooltip) tooltip(tooltipcontent);

    let closeTimeoutID = null;
    let openTimeoutID = null;
    element.on("mouseenter", () => {
        clearTimeout(closeTimeoutID);

        // Add timeout to prevent flickering of the tooltip and accidentally opening it
        openTimeoutID = setTimeout(() => {
            // Call setup function before the position is corrected
            if (preShow) preShow();

            // Correct position if it would be outside the visualization
            // space. Prioritize left over right because scrolling is
            // not supported to the left.
            // Displaying and hiding the tooltip is handled by CSS via
            // :hover
            const bound = root_element.node().getBoundingClientRect();
            const e = tooltipcontent.node().getBoundingClientRect();
            let shift = 0;
            if (e.left < bound.left) {
                shift = bound.left - e.left;
            } else if (e.right > bound.right) {
                shift = Math.max(bound.right - e.right, bound.left - e.left);
            }
            tooltipcontent.style("translate", shift + "px");

            // Scale the element if its width or height are larger than the root
            // element
            if (e.width > bound.width) {
                tooltipcontent.style("width", bound.width + "px");
            }
            if (e.height > bound.bottom - e.top) {
                tooltipcontent.style("height", (bound.bottom - e.top) + "px");
            }

            // Show tooltip
            tooltipcontent.classed("visible", true);
        }, 200);
    });
    element.on("mouseleave", () => {
        clearTimeout(openTimeoutID);

        // Hide tooltip and reset tooltip position. Add timeout to prevent 
        // flickering and accidentally closing the tooltip
        closeTimeoutID = setTimeout(() => {
            tooltipcontent.classed("visible", false)
            tooltipcontent.node().style.translate = null;
            tooltipcontent.node().style.width = null;
            tooltipcontent.node().style.height = null;
        }, 195);
    });
    return tooltipcontent;
}

function menu(element) {
    element.classed("menu-container", true);
    const m = element.append("div").classed("menu", true)
        .style("visibility", "hidden");
    element.on("click", () => {
        m.style("visibility", "visible");
    });
    window.addEventListener("mousedown", (e) => {
        if (!element.node().contains(e.target)) {
            m.style("visibility", "hidden");
        }
    });
    return m;
}


function set_url_param(key, value){
    var url = new URL(window.location.href);
    if (value) {
        url.searchParams.set(key, value);
    } else {
        url.searchParams.delete(key)
    }
    // history.pushState(null, null, url);  // update url and add to history
    history.replaceState(null, null, url);  // update url and keep history
    // window.location.replace(url);  // update url, but trigger page reload
}

class CanvasPlot {
    element;
    canvas;
    context;
    width;
    height;
    width_html;
    height_html;
    x_axis_padding;  // Padding for the x-axis in the canvas
    y_axis_padding;  // Padding for the y-axis in the canvas
    y_axis_padding_html;  // Padding for the y-axis in the HTML
    x;
    y;
    dpr;

    /**
     * Creates a canvas and axis elements to be drawn on a canvas plot.
     *
     * Width and height of the plot are determined by the `element` and can be set by CSS.
     *
     * Note: The unit "px" is only a true pixel size, if the zoom is set to
     *       100% and the the screen is not a retina display (e.g. smartphones).
     *       The canvas has the size of the number of pixels of the display to
     *       ensure sharp rendering. In html we have to use different numbers.
     */
    constructor(element, x_scale, y_scale, xAxis, yAxis, invert_y=false, layers=1) {
        this.element = element.style('position', 'relative');
        this.layers = [];
        for (let i = 0; i < layers; i++) {
            this.layers.push(this.element.append("canvas").style("width", "100%").style("height", "100%").style("position", "absolute").style("top", 0).style("left", 0));
        }
        this.canvas = this.layers[0];

        this.context = this.canvas.node().getContext("2d")
        this.xAxis = xAxis;
        this.yAxis = yAxis;
        this.invert_y = invert_y

        if (this.xAxis) this.xAxis.horizontal = true;
        if (this.yAxis) this.yAxis.horizontal = false;

        // Create plot elements
        this.x = x_scale;
        this.y = y_scale;
        this.sizeChangedListeners = [];
        this.canvasSizeChanged();


        // Track size changes of our canvas.
        new ResizeObserver(this.canvasSizeChanged.bind(this)).observe(this.element.node());
    }

    onSizeChanged(callback) {
        this.sizeChangedListeners.push(callback);
    }

    _update_dpr() {
        this.dpr = window.devicePixelRatio || 1;
        this.x_axis_padding = this.xAxis?.padding || 0;
        this.y_axis_padding = this.yAxis?.padding || 0;
        this.y_axis_padding_html = this.y_axis_padding / this.dpr;
    }

    canvasSizeChanged() {
        // Monitor the size change of the parent div, not the canvas.
        // The canvas will not shrink below canvas.height.
        // We set the canvas display to absolute so that the div can shrink

        if ((window.devicePixelRatio || 1) !== this.dpr) {
            console.log("CanvasPlot devicePixelRatio changed.", this.dpr, window.devicePixelRatio);
            this.xAxis?._update_dpr();
            this.yAxis?._update_dpr();
        }
        this._update_dpr();

        this.width_html = this.element.node().clientWidth;
        this.height_html = this.element.node().clientHeight;

        this.width = this.width_html * this.dpr;
        this.height = this.height_html * this.dpr;

        this.layers.forEach(l => {
            l.attr("width", this.width);
            l.attr("height", this.height);
            // The canvas size must match the pixel size exactly
            l.style("width", this.width_html + "px");
            l.style("height", this.height_html + "px");
        });
        this.x.range([this.y_axis_padding, this.width])
        if (this.invert_y) {
            this.y.range([0, this.height - this.x_axis_padding])
        } else {
            this.y.range([this.height - this.x_axis_padding, 0])
        }
        this.sizeChangedListeners.forEach(c => c());
    }

    drawAxes() {
        if (this.xAxis) this.xAxis.draw(this.context, this.x, this.y.range()[this.invert_y ? 1 : 0]);
        if (this.yAxis) this.yAxis.draw(this.context, this.y, this.x.range()[0]);
    }

    clear() {
        this.context.clearRect(0, 0, this.width, this.height);
    }
}

    function drawLegend(container) {
        const legend = container
            .append("div").classed("pill", true)
        for (const k of Object.keys(settings.colors)) {
            const l = legend.append("div").classed("legend-element", true)
            l.append("div").classed("legend-color", true)
                .style("background-color", settings.colors[k]);
            l.append("div").classed("legend-label", true).text(k)
        }
    }


    function drawMenu(container) {
        const menuContainer = container.append("div").classed("pill", true);
        menuContainer.append("div").html(icons['menu']);
        const m = menu(menuContainer).append("div");

        m.append("div").text("Settings").classed("menu-header", true);

        // Main plot settings
        m.append("div").classed("menu-section-label", true).text("Main Plot");
        let menuElement = m.append("div").classed("menu-element", true);
        menuElement.append("div").classed("menu-label", true).text("Font size:");
        menuElement.append("input").classed("menu-control", true).attr("type", "range").attr("min", "5").attr("max", "30").classed("slider", true).attr("step", 1).on("input", function () {
            settings.font_size = this.value;
            state.dirty[state.dirty.length - 1] = true;
            update();
        }).node().value = settings.font_size;
        menuElement = m.append("div").classed("menu-element", true)
        menuElement.append("div").classed("menu-label", true).text("Match width:");
        menuElement.append("input").classed("menu-control", true).attr("type", "range").attr("min", "1").attr("max", "90").classed("slider", true).attr("step", 1).on("input", function () {
            settings.match_width = parseInt(this.value) / 100;
            state.dirty[state.dirty.length - 1] = true;
            details_plot.precompute_utterance_positions();
            update();
        }).node().value = settings.match_width * 100;
        menuElement = m.append("div").classed("menu-element", true)
        menuElement.append("div").classed("menu-label", true).text("Show Playhead:");
        menuElement.append("input").classed("menu-control", true).attr("type", "checkbox").on("change", function () {
            settings.show_playhead = this.checked;
            state.dirty[state.dirty.length - 1] = true;
            selectedUtteranceDetails.update(state.selectedSegment);
        }).node().checked = settings.show_playhead;

        // Minimaps
        m.append("div").classed("divider", true);
        m.append("div").classed("menu-section-label", true).text("Minimaps");

        menuElement = m.append("div").classed("menu-element", true);
        menuElement.append("div").classed("menu-label", true).text("Number:")
        const num_minimaps_select = menuElement.append("select").classed("menu-control", true).on("change", function () {
            settings.minimaps.number = this.value;
            rebuild();
            set_url_param('minimaps', settings.minimaps.number);
        });
        num_minimaps_select.append("option").attr("value", 0).text("0");
        num_minimaps_select.append("option").attr("value", 1).text("1");
        num_minimaps_select.append("option").attr("value", 2).text("2");
        num_minimaps_select.append("option").attr("value", 3).text("3");

        num_minimaps_select.node().value = settings.minimaps.number;

        // const errorbar_style = container.append("div").classed("pill", true);
        // errorbar_style.append("div").classed("info-label", true).text("Error distribution");
        menuElement = m.append("div").classed("menu-element", true);
        menuElement.append("div").classed("menu-label", true).text("Error distribution:");
        const errorbar_style_select = menuElement.append("select").classed("menu-control", true).on("change", function () {
            settings.barplot.style = this.value;
            rebuild();
        });
        errorbar_style_select.append("option").attr("value", "absolute").text("Absolute");
        errorbar_style_select.append("option").attr("value", "relative").text("Relative");
        errorbar_style_select.append("option").attr("value", "hidden").text("Hidden");
        errorbar_style_select.node().value = settings.barplot.style;

        
        // const errorbar_mode = container.append("div").classed("pill", true);
        // errorbar_mode.append("div").classed("info-label", true).text("Scale exclude correct");
        // menuElement = m.append("div").classed("menu-element", true);
        // menuElement.append("div").classed("menu-label", true).text("Hi");
        // const errorbar_mode_check = menuElement.append("input").classed("menu-control", true).attr("type", "checkbox").on("change", function () {
        //     settings.barplot.scaleExcludeCorrect = this.checked;
        //     redraw();
        // });
        // errorbar_mode_check.node().checked = settings.barplot.scaleExcludeCorrect;


        // Audio
        m.append("div").classed("divider", true);
        m.append("div").classed("menu-section-label", true).text("Audio Server");
        menuElement = m.append("div").classed("menu-element", true);
        menuElement.append("div").classed("menu-label", true).text("Adress:")

        menuElement
            .append("input")
            .classed("menu-control", true)
            .attr("type", "text")
            .attr("placeholder", "e.g. http://localhost:7777")
            .on("input", function () {
                settings.audio_server = this.value;
            })
            .property("value", settings.audio_server)
        ;
    }

    function drawHelpButton(container) {
        const pill = container.append("a").attr("href", "https://github.com/fgnt/meeteval").classed("pill", true)
        pill.append('div').html(icons['help']).style("margin-right", "5px");;
        pill.append('div').text(' Help');
    }

    function drawExampleInfo(container, info) {
        const root = container;

        label = (label, value, icon=null, tooltip=null) => {
            var l = root.append("div").classed("pill", true)
            if (icon) l.append("div").classed("icon", true).html(icon);
            l.append("div").classed("info-label", true).text(label);
            l.append("div").classed("info-value", true).text(value);
            if (tooltip) addTooltip(l, tooltip);
            return l;
        }

        label("ID:", info.session_id);
        label("Length:", info.length.toFixed(2) + "s");
        if (info.system_name) label("System:", info.system_name);
        label("WER:", (info.wer.error_rate * 100).toFixed(2) + "%", null, c => {
            const wer = info.wer;
            const wer_by_speakers = info.wer_by_speakers;
            const table = c.append("table").classed("wer-table", true);
            const head = table.append("thead")
            const hr1 = head.append("tr");
            hr1.append("th");
            hr1.append("th");
            hr1.append("th");
            hr1.append("th");

            // Determine header from alignment type. If it contians orc, write by stream, otherwise, write by spekaer
            let breakdownHeader;
            if (info.alignment_type.includes("orc")) {
                breakdownHeader = "Counts by Stream";
            } else {
                breakdownHeader = "Counts by Speaker";
            }

            hr1.append("th").text(breakdownHeader).attr("colspan", Object.keys(wer_by_speakers).length).style("border-bottom", "1px solid white");

            const hr = head.append("tr")
            hr.append("th").text("");
            hr.append("th");
            hr.append("th").text("Count");
            hr.append("th").text("Relative");
            Object.keys(wer_by_speakers).forEach(speaker => { hr.append("th").text(speaker); });
            const body = table.append("tbody");
            const words = body.append("tr");
            words.append("td").text("Ref. Words");
            words.append("td");
            words.append("td").text(wer.length);
            words.append("td").text("100.0%");
            Object.entries(wer_by_speakers).forEach(([speaker, wer]) => { words.append("td").text(wer.length); });
            const correct = body.append("tr");
            correct.append("td").text("Correct");
            correct.append("td").append("div").classed("legend-color", true).style("background-color", settings.colors["correct"]);
            correct.append("td").text(wer.length - wer.substitutions - wer.deletions);
            correct.append("td").text(((wer.length - wer.substitutions - wer.deletions)/wer.length * 100).toFixed(1) + "%");
            Object.entries(wer_by_speakers).forEach(([speaker, wer]) => { correct.append("td").text(wer.length - wer.substitutions - wer.deletions); });
            const substitution = body.append("tr");
            substitution.append("td").text("Substitution");
            substitution.append("td").append("div").classed("legend-color", true).style("background-color", settings.colors["substitution"]);
            substitution.append("td").text(wer.substitutions);
            substitution.append("td").text((wer.substitutions / wer.length * 100).toFixed(1) + "%");
            Object.entries(wer_by_speakers).forEach(([speaker, wer]) => { substitution.append("td").text(wer.substitutions); });
            const insertion = body.append("tr");
            insertion.append("td").text("Insertion");
            insertion.append("td").append("div").classed("legend-color", true).style("background-color", settings.colors["insertion"]);
            insertion.append("td").text(wer.insertions);
            insertion.append("td").text((wer.insertions / wer.length * 100).toFixed(1) + "%");
            Object.entries(wer_by_speakers).forEach(([speaker, wer]) => { insertion.append("td").text(wer.insertions); });
            const deletion = body.append("tr");
            deletion.append("td").text("Deletion");
            deletion.append("td").append("div").classed("legend-color", true).style("background-color", settings.colors["deletion"]);
            deletion.append("td").text(wer.deletions);
            deletion.append("td").text((wer.deletions / wer.length * 100).toFixed(1) + "%");
            Object.entries(wer_by_speakers).forEach(([speaker, wer]) => { deletion.append("td").text(wer.deletions); });
            c.append("div").classed("tooltip-info", true).text("Note: Values don't add up to 100% (except when Insertion=0)\nRef. Words = Correct + Substitution + Deletion\nHyp. Words = Correct + Substitution + Insertion");
        });
        label("Alignment:", info.alignment_type, null,
            c => c.append('div').classed('wrap-60', true).html("The alignment algorithm used to generate this visualization. Available are:" +
            "<ul>" +
            "<li><code>cp</code>: concatenated minimum-permutation</li>" +
            "<li><code>tcp</code>: time-constrained minimum permutation</li>" +
            "<li><code>orc</code>: (speaker-agnostic) optimal reference combination</li>" +
            "<li><code>tcorc</code>: (speaker-agnostic) time-constrained optimal reference combination</li>" +
            "</ul>" +
            "<p>All visualizations are generated with <code>reference_sort='segment'</code> and <code>hypothesis_sort='segment'</code>. " +
            "Time-constrained alignments are generated with <code>collar=5</code>, <code>ref_pseudo_word_level_timing='character_based'</code> and " +
            "<code>hyp_pseudo_word_level_timing='character_based_points'</code>. " +
            "Word lengths for the visualization are determined with the <code>'character_based'</code> strategy.</p>" +
            "<p>This setting has to be selected when generating the visualization. " +
            "Check the documentation for details.</p>"
        )
        )
        if (info.wer.reference_self_overlap?.overlap_rate) label(
            "Reference self-overlap:",
            (info.wer.reference_self_overlap.overlap_rate * 100).toFixed(2) + "%",
            icons["warning"],
            c => {
                c.append('div').classed('wrap-40', true).text("Self-overlap is the percentage of time that a speaker annotation overlaps with itself. " +
            "On the reference, this is usually an indication for annotation errors.\n" +
            "Extreme self-overlap can lead to unexpected WERs!");
                const d = c.append('div').classed("menu-element", true).style("margin-top", "1em");
                d.append('div').text("< Show previous").classed("clickable", true)
                    .on("click", () => selectNextMatchingSegment(u => u.utterance_overlaps.length > 0 && u.source == 'reference', true, true)); 
                d.append('div').text("Show next >").classed("clickable", true).style("margin-left", "auto")
                    .on("click", () => selectNextMatchingSegment(u => u.utterance_overlaps.length > 0 && u.source == 'reference', true, false));
                }
        ).classed("warn", true);
        if (info.wer.hypothesis_self_overlap?.overlap_rate) label(
            "Hypothesis self-overlap:",
            (info.wer.hypothesis_self_overlap.overlap_rate * 100).toFixed(2) + "%",
            icons["warning"],
            c => {
                c.append('div').classed('wrap-40', true).text("Self-overlap is the percentage of time that a speaker annotation overlaps with itself. " +
            "On the hypothesis, this often indicates systematic errors.\n" +
            "Extreme self-overlap can lead to unexpected WERs!")
               const d = c.append('div').classed("menu-element", true).style("margin-top", "1em");
                d.append('div').text("< Show previous").classed("clickable", true)
                    .on("click", () => selectNextMatchingSegment(u => u.utterance_overlaps.length > 0 && u.source == 'hypothesis', true, true)); 
                d.append('div').text("Show next >").classed("clickable", true).style("margin-left", "auto")
                    .on("click", () => selectNextMatchingSegment(u => u.utterance_overlaps.length > 0 && u.source == 'hypothesis', true, false));
                }
        ).classed("warn", true);
    }


    /**
     * Search bar component. Modifies "words" in-place by setting the `.highlight`
     * attribute of matching words to `true`.
     */
    class SearchBar {
        constructor(container, state, initial_query) {
            this.last_search = "";
            this.num_matches = 0;
            this.state = state;
            this.container = container.append("div").classed("pill", true).classed("search-bar", true);
            this.text_input = this.container.append("input").attr("type", "text").attr("placeholder", "Regex (e.g., s?he)...");
            this.matched_words_indices = []; // Indices of words that match the regex needed for jumping between matches
            this.current_focus_index = null;    // The currently focused word, indexing into matched_words_indices

            if (initial_query) this.text_input.node().value = initial_query;

            // Start search when clicking on the button
            this.match_number = this.container.append("div").classed("match-number", true);
            this.search_button = this.container.append("button").text("Search").on("click", () => this.search(this.text_input.node().value));
            this.prev_button = this.container.append("button").text("<").on("click", () => this.search(this.text_input.node().value, true));
            this.next_button = this.container.append("button").text(">").on("click", () => this.search(this.text_input.node().value, false));

            // Start search on Enter
            this.text_input.on("keydown", (event) => {
                if (event.key === "Enter") {
                    this.search(this.text_input.node().value, event.shiftKey);
                }
            });
        }

        search(regex, reverse=false) {
            if (regex !== this.last_search) {
                this.last_search = regex;
                if (this.current_focus_index !== null) {
                    this.state.words[this.matched_words_indices[this.current_focus_index]].focused = false;
                }
                this.matched_words_indices = [];
                this.current_focus_index = null;

                // Test all words against the regex. Use ^ and $ to get full match
                this.num_matches = 0;
                data.utterances.forEach(u => u.highlight = false);
                if (regex === "")  {
                    this.state.words.forEach(w => w.highlight = false);
                } else {
                    const re = new RegExp("^" + regex + "$", "i");
                    for (const [index, w] of this.state.words.entries()) {
                        w.highlight = re.test(w.words);
                        if (w.highlight) {
                            this.matched_words_indices.push(index);
                            data.utterances[w.utterance_index].highlight = true;
                            this.num_matches++;
                        }
                    } 
                }

                // Adjust UI: enable buttons and show number of matches
                if (this.num_matches > 0) {
                    this.prev_button.attr("disabled", null);
                    this.next_button.attr("disabled", null);
                    this.match_number.text(`(${this.num_matches})`);
                } else {
                    this.prev_button.attr("disabled", true);
                    this.next_button.attr("disabled", true);
                    this.match_number.text("");
                }

                // Update state
                this.state.dirty.fill(true);
                update();

                // Update URL
                set_url_param('regex', regex)
            } else {
                // Select the first/next occurrence, but only on second hit of the search button /
                // enter key. We don't want to change the selection immediately
                if (this.num_matches > 0) {
                    if (this.current_focus_index === null) {
                        if (reverse) this.current_focus_index = this.matched_words_indices.length - 1;
                        else this.current_focus_index = 0;
                    } else {
                        this.state.words[this.matched_words_indices[this.current_focus_index]].focused = false;
                        this.current_focus_index = this.current_focus_index + (reverse ? -1 : 1);
                    }
                    if (this.current_focus_index < 0 || this.current_focus_index > this.matched_words_indices.length - 1){
                        this.current_focus_index = null;
                    }
                    if (this.current_focus_index !== null) {
                        const word_index = this.matched_words_indices[this.current_focus_index];
                        const word = this.state.words[word_index];
                        word.focused = true;

                        // Move such that the word is in the center of the view area
                        // Make sure that the word highlight is drawn
                        state.dirty[state.dirty.length - 1] = true;
                        focusPoint((word.start_time + word.end_time) / 2, false);
                        selectSegment(data.utterances[word.utterance_index]);
                        update();   // Only necessary if view area is not moved and the segment selection is not changed
                    }
                }
            }
        }
    }

    class ErrorBarPlot {
        constructor(canvas_plot, num_bins, style='absolute', scaleExcludeCorrect=false) {
            this.plot = canvas_plot;
            this.bin = d3.bin().thresholds(200).value(d => (d.start_time + d.end_time) / 2)
            this.max = 0;
            this.binned_words = [];
            this.style = style;
            this.scaleExcludeCorrect = scaleExcludeCorrect;
            this.plot.onSizeChanged(this.draw.bind(this));
        }

        updateBins(words, viewArea) {
            var bin_max = 0;
            const self = this;

            this.binned_words = this.bin.domain(viewArea)(words).map(d => {
                // This is for utterances
                // d.substitutions = d.map(i => i.substitutions).reduce((a, b) => a + b, 0);
                // d.insertions = d.map(i => i.insertions || 0).reduce((a, b) => a + b, 0);
                // d.deletions = d.map(i => i.deletions || 0).reduce((a, b) => a + b, 0);
                // d.total = d.map(i => i.total || 0).reduce((a, b) => a + b, 0);
                d.substitutions = d.flatMap(w => w.matches).map(m => !!m && m[1] === 'substitution').reduce((a, b) => a + b, 0);
                d.insertions = d.flatMap(w => w.matches).map(m => !!m && m[1] === 'insertion').reduce((a, b) => a + b, 0);
                d.deletions = d.flatMap(w => w.matches).map(m => !!m && m[1] === 'deletion').reduce((a, b) => a + b, 0);
                d.total = d.flatMap(w => w.matches).map(m => !!m && m[1] !== 'ignored').reduce((a, b) => a + b, 0);

                // d.insertions = d.map(w => w.match_type === 'insertion').reduce((a, b) => a + b, 0);
                // d.deletions = d.map(w => w.match_type === 'deletion').reduce((a, b) => a + b, 0);
                // d.total = d.map(w => w.match_type !== 'ignored').reduce((a, b) => a + b, 0);
                d.highlight = d.map(w => w.highlight).reduce((a, b) => a || b, false);

                // Compute relative numbers if requested
                if (self.style === 'relative') {
                    d.substitutions = d.substitutions / (d.total || d.substitutions || 1);
                    d.insertions = d.insertions / (d.total || d.insertions || 1);
                    d.deletions = d.deletions / (d.total || d.deletions || 1);
                    d.total = 1;
                }

                // Compute upper limit of plot
                if (self.scaleExcludeCorrect) {
                    bin_max = Math.max(bin_max, d.substitutions + d.insertions + d.deletions);
                } else {
                    bin_max = Math.max(bin_max, d.total, d.substitutions + d.insertions + d.deletions);
                }
                return d
            });
            this.max = bin_max;
            this.plot.y.domain([0, bin_max * 1.1]);
        }

        update(viewArea, filteredWords) {
            this.plot.x.domain(viewArea);
            this.words = filteredWords;
            this.updateBins(filteredWords, viewArea);
        }

        drawBars() {
            const self = this;
            this.binned_words.forEach(b => {
                const x = this.plot.x(b.x0);
                const width = this.plot.x(b.x1) - this.plot.x(b.x0);
                var y = 0;
                const bottom = this.plot.y(0);
                y = (y) => this.plot.y(y) - bottom;

                if (b.highlight) {
                    height = 3;
                    this.plot.context.beginPath();
                    this.plot.context.fillStyle=settings.colors.highlight;
                    this.plot.context.rect(x, bottom, width, this.plot.y(this.plot.y.domain()[1]) - bottom);
                    this.plot.context.stroke();
                    this.plot.context.fill();
                }

                if (self.style === 'absolute') {
                    this.plot.context.strokeStyle = "gray";
                    this.plot.context.fillStyle = settings.colors["correct"];
                    this.plot.context.beginPath();
                    this.plot.context.rect(x, bottom, width, y(b.total));
                    this.plot.context.stroke();
                    this.plot.context.fill();
                }

                // Substitutions
                var height = y(b.substitutions);
                this.plot.context.fillStyle = settings.colors["substitution"];
                this.plot.context.fillRect(x, bottom, width, height)
                var bottom_ = bottom + height;

                // Insertions
                height = y(b.insertions);
                this.plot.context.fillStyle = settings.colors["insertion"];
                this.plot.context.fillRect(x, bottom_, width, height)
                bottom_ = bottom_ + height;

                // Deletions
                height = y(b.deletions);
                this.plot.context.fillStyle = settings.colors["deletion"];
                this.plot.context.fillRect(x, bottom_, width, height)
                bottom_ = bottom_ + height;
            });
        }

        draw() {
            this.plot.clear();
            this.drawBars();
            this.plot.drawAxes();
        }
    }

    class WordPlot {
        constructor(plot) {
            this.plot = plot;
            this.words = [];
            this.plot.onSizeChanged(this.draw.bind(this));
            this.on_zoom_to_callbacks = [];
        }

        drawWords() {
            const [begin, end] = this.plot.x.domain();
            this.plot.context.strokeStyle = "gray";
            const bandwidth = this.plot.y.bandwidth() / 2;
            this.words.forEach(w => {
                this.plot.context.beginPath();
                let y_ = this.plot.y(w.speaker);
                if (w.source === "hypothesis") y_ += bandwidth;
                this.plot.context.rect(
                    this.plot.x(w.start_time),
                    y_,
                    this.plot.x(w.end_time) - this.plot.x(w.start_time),
                    bandwidth,
                );
                if (w.highlight) {
                    this.plot.context.fillStyle = settings.colors.highlight;
                    this.plot.context.fill()
                } else if (w.matches) {
                    this.plot.context.fillStyle = settings.colors[w.matches[0][1]];
                    this.plot.context.fill();
                } else {
                    this.plot.context.stroke();
                }
            })
        }

        update(viewArea, filteredWords) {
            this.plot.x.domain(viewArea);
            this.words = filteredWords;
        }

        draw() {
            this.plot.clear();
            this.drawWords();
            this.plot.drawAxes();
        }
    }

    class Minimap {
        constructor(element, state, index) {
            const e = element.classed("plot minimap", true).style("height", '90px')
            this.state = state
            this.index = index

            const x_scale = d3.scaleLinear().domain(state.viewAreas[index]);

            if (settings.barplot.style !== "hidden") {
                this.error_bars = new ErrorBarPlot(
                    new CanvasPlot(
                        e.append('div').style('height', '30%'),
                        x_scale,
                        d3.scaleLinear().domain([1, 0]),
                        null,
                        new Axis(50, 3),
                ), 200, settings.barplot.style, settings.barplot.scaleExcludeCorrect);
                this.error_bars.plot.element.append("div").classed("plot-label", true).style("margin-left", this.error_bars.plot.y_axis_padding_html + "px").text("Error distribution");
            }
            this.word_plot = new WordPlot(
                new CanvasPlot(
                    e.append('div').style('height',this.error_bars ? '70%' : '100%'),
                    x_scale,
                    state.speakerScale,
                    new CompactAxis(10, "time"), new Axis(50), true),
            );
            this.word_plot.plot.element.append("div").classed("plot-label", true).style("margin-left", this.word_plot.plot.y_axis_padding_html + "px").text("Segments");

            // Setup brush
            this.svg = e.append("svg")
                .style("position", "absolute").style("top", 0).style("left", 0).style("width", "100%").style("height", "100%");

            this.brush = d3.brushX()
                .extent([
                    [
                        Math.max(this.error_bars?.plot.y_axis_padding_html || 0, this.word_plot.plot.y_axis_padding_html),
                        0
                    ],
                    [this.word_plot.plot.width_html, this.word_plot.plot.height_html + (this.error_bars?.plot.height_html || 0)]])
                .on("brush", this._onselect.bind(this))
                .on("end", this._onselect.bind(this))
                .touchable(() => true); // Required for touch support for chrome on Laptops

            this.brush_group = this.svg.append("g")
                .attr("class", "brush")
                .call(this.brush);

            // Redraw brush when size changes. This is required because the brush range / extent will otherwise keep the old value (in screen size)
            this.word_plot.plot.onSizeChanged(() => {
                 // This seems hacky, but I didn't find another way to modify the height of the brush
                const height = this.word_plot.plot.height_html + (this.error_bars?.plot.height_html || 0);
                this.brush.extent([
                    [
                        Math.max(this.error_bars?.plot.y_axis_padding_html || 0, this.word_plot.plot.y_axis_padding_html),
                        0
                    ],
                    [this.word_plot.plot.width_html, height]]
                );
                // this.brush.extent modifies the overlay rect, but not the selection rect. We have to set the
                // selection rect manually
                this.brush_group.property('__brush').extent[1][1] = height;
                if ( this.brush_group.property('__brush').selection) {
                    this.brush_group.property('__brush').selection[1][1] = height;
                }
                // Re-build brush svg. The SVG size and drag area can break without this.
                this.brush_group.call(this.brush);

                // Re-draw plot and re-position brush
                this.updateViewArea();
                this.updateBrush();
            });

            // Make the minimap resizable
            const resize_handle = e.append('div').classed('minimap-resize-handle', true);
            drag(resize_handle, e => {
                const parent_top = element.node().getBoundingClientRect().top;
                const new_height = Math.max(e.clientY - parent_top - 2/*half of resize-handle height*/, 20);
                element.style('height', new_height + "px");
            }, () => resize_handle.classed('active', true), () => resize_handle.classed('active', false));
            resize_handle.on("touchstart", () => resize_handle.classed('active', true));
            resize_handle.on("touchend", () => resize_handle.classed('active', false));
            resize_handle.on("touchmove", (e) => {
                // Select the first touch that started in the touch handle. Any
                // further touches are ignored.
                const touch = e.targetTouches[0];
                const parent_top = element.node().getBoundingClientRect().top;
                const new_height = Math.max(touch.clientY - parent_top - 2/*half of resize-handle height*/, 20);
                element.style('height', new_height + "px");
            })

            this.updating = false;

            this.drawTracker = {}
        }

        draw() {
            if (this.error_bars) this.error_bars.draw();
            this.word_plot.draw();
        }

        updateViewArea() {
            const viewArea = this.state.viewAreas[this.index];
            const words = this.state.filteredWords[this.index];

            // Update sub-plots
            if (this.error_bars) this.error_bars.update(viewArea, words);
            this.word_plot.update(viewArea, words);
            this.draw();
        }

        updateBrush() {
            const viewArea = this.state.viewAreas[this.index];
            const brushArea = this.state.viewAreas[this.index + 1];
            // Prevent update loops with brush
            this.updating = true;
            // brush area is greater than view area
            if (!brushArea || brushExceedsViewArea(viewArea, brushArea)) {
                // Remove brush.
                // !brushArea is a bit hacky, but we get a sizeChanged event
                // when the minimap is removed, in which case the brushArea
                // becomes null.
                this.brush_group.call(this.brush.move, null);
            } else {
                this.brush_group.call(this.brush.move, [
                    this.word_plot.plot.x(brushArea[0]) / this.word_plot.plot.dpr,
                    this.word_plot.plot.x(brushArea[1]) / this.word_plot.plot.dpr,
                ])
            }
            this.updating = false;
        }

        _onselect(event) {
            if (this.updating) return;
            let selectionDomain;
            if (event.selection) {
                selectionDomain = event.selection.map((brush_pos) => {
                    return this.word_plot.plot.x.invert(brush_pos * this.word_plot.plot.dpr)
                });
            } else {
                selectionDomain = this.state.viewAreas[this.index];
            }
            updateViewArea(this.index + 1, selectionDomain);
        }
    }


    class DetailsPlot {
        constructor(plot, state, utterances, markers, initial=null) {
            this.plot = plot;
            this.plot.element.classed("plot", true);

            this.state = state;
            this.utterances = utterances;
            this.filtered_utterances = utterances;
            this.max_domain = plot.y.domain();
            this.markers = markers;
            this.filtered_markers = markers;

            // Precompute alignment / stitches for insertions/substitutions
            this.matches = state.words.flatMap((w) => {
                if (!w.matches) return [];
                return w.matches
                    .filter(m => m[0])  // Only look at matches between words
                    .map(m => {
                        const other = state.words[m[0]];
                        const [left, right] = w.source === "hypothesis" ? [other, w] : [w, other];
                        return {
                            speaker: w.speaker,
                            match_type: m[1],
                            left_center_time: left.center_time,
                            right_center_time: right.center_time,
                            start_time: Math.min(left.center_time, right.center_time),
                            end_time: Math.max(left.center_time, right.center_time),
                            left_utterance: utterances[left.utterance_index],
                            right_utterance: utterances[right.utterance_index],
                        }
                    })
            });
            this.filtered_matches = this.matches;

            // Precompute utterance x positions and widths
            this.precompute_utterance_positions = () => {
                const match_width = settings.match_width * this.plot.x.bandwidth() / 2;
                const columnwidth = this.plot.x.bandwidth() / 2 - match_width;

                this.utterances.forEach(u => {
                    let x = this.plot.x(u.speaker);
                    let width = columnwidth;

                    if (u.source === "hypothesis") {
                        x += this.plot.x.bandwidth() / 2 + match_width;
                    }

                    if (u.utterance_overlaps) {
                        width = columnwidth * u.overlap_width;
                        x = x + width * u.overlap_shift;
                        width = width - 2*constants.utteranceMarkerOverhang;
                    }
                    u.x = x;
                    u.width = width;
                })
                
            };
            this.plot.onSizeChanged(this.precompute_utterance_positions);

            this.utteranceSelectListeners = [];

            // Plot label
            this.plot.element.append("div").classed("plot-label", true).style("margin-left", this.plot.y_axis_padding_html + "px").text("Detailed matching");

            // Click handler for selecting utterances
            const self = this;
            this.plot.element.on("click", (event) => {
                const screenX = event.layerX * self.plot.dpr;  // convert from html px to canvas px
                const screenY = event.layerY * self.plot.dpr;  // convert from html px to canvas px
                const y = self.plot.y.invert(screenY);

                // Brute force go through all utterances and check if the click is inside
                // Use the precomputed x and width values
                // This should be fast enough since this.filtered_utterances contains only the visible utterances
                // and this action is not performed frequently
                const utterance_candidates = this.filtered_utterances.filter(
                    u => u.start_time < y && u.end_time > y && u.x <= screenX && u.x + u.width >= screenX
                )
                if (utterance_candidates.length > 0) {
                    console.log('click')
                    // Select the utterance that was clicked on. Move view to utterance on double click
                    selectSegment(utterance_candidates[0], event.detail === 2);

                    // With the current layout, utterances should never overlap.
                    // Log a warning if this happens
                    if (utterance_candidates.length > 1) console.warn("Multiple utterances selected. This should not happen.")
                }
                else selectSegment(null);
            })

            // Scrolling with a mouse wheel
            this.wheel_tracker = {}
            let deltaY = 0;
            let hitCount = 0;
            this.plot.element.on("wheel", (event) => {
                // Collate multiple wheel events as one "big" wheel event.
                // 5 milliseconds aren't noticeable, but prevent freezing from free rolling mouse wheels
                deltaY += event.deltaY;
                hitCount += 1;
                event.preventDefault();
                call_throttled(() => {
                    let [begin, end] = this.plot.y.domain();
                    let delta = deltaY;
                    deltaY = 0;
                    delta = (this.plot.y.invert(delta) - this.plot.y.invert(0)) * 0.5   // TODO: magic number
                    if (hitCount > 1)
                        console.log('High frequently appearing wheel events. Group', hitCount, 'events together.')
                    hitCount = 0;

                    if (event.ctrlKey) {
                        // Zoom when ctrl is pressed. Zoom centered on mouse position
                        const mouse_y = this.plot.y.invert(event.layerY);
                        const ratio = (mouse_y - begin) / (end - begin);
                        begin -= delta * ratio;
                        end += delta * (1 - ratio);
                    } else {
                        begin = begin + delta;
                        end = end + delta;
                    }
                    updateViewArea(this.state.viewAreas.length - 1, [begin, end]);
                }, this.wheel_tracker, 25)
            }, false)


            // Use a state to support concurrent mousemove events
            let drag_state = {};
            drag(this.plot.element, (e, delta_y) => {
                if (delta_y){
                    drag_state.delta_y += delta_y;
                    const delta = this.plot.y.invert(drag_state.delta_y * this.plot.dpr) - this.plot.y.invert(0);
                    updateViewArea(this.state.viewAreas.length - 1, [drag_state.begin - delta, drag_state.end - delta]);
                }
            }, () => {
                let [begin, end] = this.plot.y.domain();
                drag_state.begin = begin;
                drag_state.end = end;
                drag_state.delta_y = 0;
            });

            var startTouch = {};

            const touchstart = (event) => {
                // TouchList doesn't implement iterator
                startTouch.Y = [];
                for (let i = 0; i < event.targetTouches.length; i++) {
                    startTouch.Y.push(event.targetTouches[i].clientY * this.plot.dpr);
                }
                [startTouch.begin, startTouch.end] = this.plot.y.domain();
            }

            this.plot.element.on("touchstart", touchstart);
            this.plot.element.on("touchend", touchstart);

            this.plot.element.on("touchmove", event => {
                // This can happen when a touch move is started outside of the
                // element and then moved into the element, but the starting
                // element does not cancel the event. We don't want to mix
                // multiple touch handlers
                if (event.cancelable === false) return;

                // TouchList doesn't implement iterator
                var touchY = [];
                for (let i = 0; i < event.targetTouches.length; i++) {
                    touchY.push(event.targetTouches[i].clientY * this.plot.dpr);
                }
                if (startTouch.Y) {
                    // Use the delta between the touches that are furthest apart
                    const top = this.plot.element.node().getBoundingClientRect().top;
                    const minY = Math.min(...touchY) - top;
                    const maxY = Math.max(...touchY) - top;
                    const lastMinY = Math.min(...startTouch.Y) - top;
                    const lastMaxY = Math.max(...startTouch.Y) - top;
                    let [begin, end] = [startTouch.begin, startTouch.end];

                    if (lastMaxY - lastMinY > 0 && maxY - minY > 0) {
                         // At least two touch points. Zoom and move
                        const newBegin = begin + (end - begin) * (lastMinY*maxY - lastMaxY*minY) / (this.plot.height * (maxY - minY));
                        end = (this.plot.height / minY - lastMinY / minY) * begin + lastMinY / minY * end + (1 - this.plot.height / minY) * newBegin;
                        begin = newBegin;
                    } else {
                        // Only one touch point
                        const center = this.plot.y.invert((maxY + minY) / 2);
                        const lastCenter = this.plot.y.invert((lastMaxY + lastMinY) / 2);
                        const delta = lastCenter - center;
                        begin += delta;
                        end += delta;
                    }

                    updateViewArea(this.state.viewAreas.length - 1, [begin, end]);
                    event.preventDefault()
                }
            })

            this.plot.onSizeChanged(this.draw.bind(this));
        }

        updatePlayhead(time) {
            this.playhead = time;
            this.drawPlayhead();
        }

        drawPlayhead() {
            if (!settings.show_playhead) return;
            const context = this.plot.layers[1].node().getContext('2d');
            context.clearRect(0, 0, this.plot.width, this.plot.height);
            // Draw horizontal line at playhead position
            context.strokeStyle = "green";
            context.lineWidth = 2;
            context.beginPath();
            context.moveTo(0, this.plot.y(this.playhead));
            context.lineTo(this.plot.width, this.plot.y(this.playhead));
            context.stroke();
        }

        drawDetails() {
            const filtered_words = this.filtered_words;

            // Draw help message and exit if the amount of displayed words is too high
            // This would lead to a very slow rendering and any information would be lost due to the scale
            if (filtered_words.length > 3000) {
                this.plot.context.font = `${30 * this.plot.dpr}px Arial`;
                this.plot.context.textAlign = "center";
                this.plot.context.textBaseline = "middle";
                this.plot.context.fillStyle = "gray";
                this.plot.context.fillText("Zoom in or select a smaller region in the minimap above", this.plot.width / 2, this.plot.height / 2);
                return;
            }

            // Precompute constants required later
            const filtered_utterances = this.filtered_utterances;
            const context = this.plot.context;
            const draw_text = filtered_words.length < 400;
            const draw_boxes = filtered_words.length < 1000;
            const draw_utterance_markers = filtered_words.length < 2000;
            const match_width = settings.match_width * this.plot.x.bandwidth() / 2;
            const stitch_offset = Math.min(constants.minStitchOffset, match_width / 2);
            const rectwidth = this.plot.x.bandwidth() / 2 - match_width;

            // Draw background: Gray bands for each speaker
            const min_y = this.plot.y.range()[0];
            const plot_height = this.plot.y.range()[1] - this.plot.y.range()[0];
            const width = this.plot.x.bandwidth();
            for (let i = 0; i < this.plot.x.domain().length; i++) {
                const speaker = this.plot.x.domain()[i];
                const x = this.plot.x(speaker);
                context.fillStyle = "#eee";
                context.fillRect(x, min_y, width, plot_height);
            }

            // Draw red lines in the background where the the selected segment
            // starts and ends
            if (draw_utterance_markers && this.state.selectedSegment) {
                const [minX, maxX] = this.plot.x.range();
                context.lineWidth = .5;
                context.strokeStyle = 'red';

                // Start point
                var y = this.plot.y(this.state.selectedSegment.start_time) - 1;
                context.beginPath();
                context.moveTo(minX, y);
                context.lineTo(maxX, y);

                // End point
                y = this.plot.y(this.state.selectedSegment.end_time) + 1;
                context.moveTo(minX, y);
                context.lineTo(maxX, y);
                context.stroke();
            }

            // Draw markers. This feature is not yet fully supported
            const filtered_markers = this.filtered_markers;
            // Draw a range marker on the left side of the plot with two lines spanning the full width
            if (filtered_markers) filtered_markers.forEach(m => {
                // TODO: custom color and label, markers for speakers
                if (m.type === "range") {
                    const y0 = this.plot.y(m.start_time);
                    const y1 = this.plot.y(m.end_time);
                    context.fillStyle = m.color ?? "purple";
                    context.strokeStyle = m.color ?? "purple";
                    context.lineWidth = .5;
                    context.fillRect(this.plot.y_axis_padding, y0, 10, y1 - y0);
                    context.beginPath();
                    context.moveTo(this.plot.y_axis_padding, y0);
                    context.lineTo(this.plot.x.range()[1], y0);
                    context.moveTo(this.plot.y_axis_padding, y1);
                    context.lineTo(this.plot.x.range()[1], y1);
                    context.stroke();
                } else if (m.type === "point") {
                    const y = this.plot.y(m.time);
                    context.fillStyle = m.color ?? "purple";
                    context.strokeStyle = m.color ?? "purple";
                    context.lineWidth = .5;
                    context.beginPath();
                    context.arc(this.plot.y_axis_padding + 5, y, 5, 0, 2 * Math.PI);
                    context.moveTo(this.plot.y_axis_padding, y);
                    context.lineTo(this.plot.x.range()[1], y);
                    context.fill();
                    context.stroke();
                }
            });

            // Draw word boxes
            filtered_words.forEach(d => {
                // Compute the actual horizontal position and width of the box
                // considering overlaps with other utterances
                const utterance = this.utterances[d['utterance_index']];

                // Draw word boxes
                context.beginPath();
                context.rect(
                    utterance.x,
                    this.plot.y(d.start_time),
                    utterance.width,
                    this.plot.y(d.end_time) - this.plot.y(d.start_time)
                );

                // Fill box with match color
                if (d.matches?.length > 0) {
                    context.fillStyle = settings.colors[d.matches[0][1]];
                    context.fill();
                }

                // Draw inner box border with highlight color
                if (d.highlight && d.focused){
                    context.save();
                    context.clip(); // Clip to the box so that it doesn't overlap with other words
                    context.strokeStyle = settings.colors.highlight;
                    context.lineWidth = 20;
                    context.stroke();
                    context.restore();
                }

                // Draw box border
                if (d.matches?.length > 0) {
                    context.strokeStyle = "gray";
                    context.lineWidth = 2;
                    if (draw_boxes) context.stroke();
                }

                // Draw (stub) stitches for insertion / deletion
                // These do not connect to other words, but are drawn as a straight line
                // ending in the space between reference and hypothesis
                if (d.matches?.length > 0) {
                    const [match_index, match_type] = d.matches[0];
                    context.beginPath();
                    context.lineWidth = 2;
                    context.strokeStyle = settings.colors[match_type];
                    if (match_type === 'insertion') {
                        const y = this.plot.y(d.center_time);
                        context.moveTo(utterance.x, y);
                        context.lineTo(utterance.x - stitch_offset, y);
                    } else if (match_type === 'deletion') {
                        const y = this.plot.y(d.center_time);
                        context.moveTo(utterance.x + utterance.width, y);
                        context.lineTo(utterance.x + utterance.width + stitch_offset, y);
                    }
                    context.stroke();
                }
            });

            // Draw stitches for correct match / substitution
            // TODO: precompute and draw ins/del matches here as well
            this.filtered_matches.forEach(m => {
                // Substitution or correct
                context.beginPath();
                const bandleft = this.plot.x(m.speaker);
                context.strokeStyle = settings.colors[m.match_type];
                context.moveTo(m.left_utterance.x + m.left_utterance.width, this.plot.y(m.left_center_time));
                context.lineTo(bandleft + rectwidth + stitch_offset, this.plot.y(m.left_center_time));
                context.lineTo(bandleft + rectwidth + 2 * match_width - stitch_offset, this.plot.y(m.right_center_time));
                context.lineTo(m.right_utterance.x, this.plot.y(m.right_center_time));
                context.stroke();
            });

            // Draw word text
            // This is done after the stitches so that the text is on top even if stitches or boxes overlap
            context.font = `${settings.font_size * this.plot.dpr}px Arial`;
            context.textAlign = "center";
            context.textBaseline = "middle";
            context.lineWidth = 1;

            if (draw_text) filtered_words.forEach(d => {
                const utterance = this.utterances[d['utterance_index']];
                let x = utterance.x + utterance.width / 2;  // Center of the utterance
                let y_ = this.plot.y((d.start_time + d.end_time) / 2);

                if (d.highlight) {
                    // Color text background with highlight color
                    const textMetrics = context.measureText(d.words);
                    context.fillStyle = settings.colors.highlight;
                    context.fillRect(
                        x - textMetrics.width / 2 - 3,
                        y_ - (textMetrics.fontBoundingBoxAscent + textMetrics.fontBoundingBoxDescent) / 2,
                        textMetrics.width + 6,
                        textMetrics.fontBoundingBoxAscent + textMetrics.fontBoundingBoxDescent
                    );
                }

                if (d.matches === undefined) context.fillStyle = "gray";
                else context.fillStyle = '#000';

                context.fillText(d.words, x, y_);
            })

            // Draw utterance begin and end markers
            // This is done after drawing the word boxes so that the markers are visible
            // even when the word boxes are too crammed
            const markerDepth = constants.utteranceMarkerDepth;
            const markerOverhang = constants.utteranceMarkerOverhang;
            if (draw_utterance_markers) filtered_utterances.forEach(d => {
                context.strokeStyle = "black";
                context.lineWidth = 1.5;
                context.beginPath();

                // x is the left side of the marker
                const x = d.x;
                const width = d.width;

                // Begin marker
                var y = this.plot.y(d.start_time) - 1;
                context.moveTo(x - markerOverhang, y + markerDepth);
                context.lineTo(x - markerOverhang, y);
                context.lineTo(x + width + markerOverhang, y);
                context.lineTo(x + width + markerOverhang, y + markerDepth);

                // End marker
                y = this.plot.y(d.end_time) + 1;
                context.moveTo(x - markerOverhang, y - markerDepth);
                context.lineTo(x - markerOverhang, y);
                context.lineTo(x + width + markerOverhang, y);
                context.lineTo(x + width + markerOverhang, y - markerDepth);
                context.stroke();

                // Draw marker that text is empty
                if (d.words === "" && draw_text) {
                    context.beginPath();
                    context.textAlign = "center";
                    context.textBaseline = "middle";
                    context.strokeStyle = "lightgray";
                    context.linewidth = 1;
                    const x_ = x + d.width / 2;
                    context.font = `italic ${settings.font_size * this.plot.dpr}px Arial`;
                    context.fillStyle = "gray";
                    context.fillText('(empty)', x_, (this.plot.y(d.start_time) + this.plot.y(d.end_time)) / 2);
                }
            });

            // Draw boundary around the selected utterance
            if (this.state.selectedSegment) {
                const d = this.state.selectedSegment;
                context.beginPath();
                context.strokeStyle = "red";
                context.lineWidth = 3;
                context.rect(d.x, this.plot.y(d.start_time), d.width, this.plot.y(d.end_time) - this.plot.y(d.start_time));
                context.stroke();

                // Write begin time above begin marker
                context.font = `italic ${settings.font_size * this.plot.dpr}px Arial`;
                context.fillStyle = "gray";
                context.textAlign = "center";
                context.textBaseline = "bottom";
                {
                    const text = `begin time: ${d.start_time.toFixed(2)}`;
                    const textMetrics = context.measureText(text);
                    context.fillStyle = "#eee";
                    context.fillRect(d.x + d.width / 2 - textMetrics.width / 2 - 3, this.plot.y(d.start_time), textMetrics.width + 6, - (textMetrics.fontBoundingBoxAscent + textMetrics.fontBoundingBoxDescent));
                    context.fillStyle = "gray";
                    context.fillText(text, d.x + d.width / 2, this.plot.y(d.start_time) - 1);
                }

                // Write end time below end marker
                {
                    context.textBaseline = "top";
                    const text = `end time: ${d.end_time.toFixed(2)}`;
                    const textMetrics = context.measureText(text);
                    context.fillStyle = "#eee";
                    context.fillRect(d.x + d.width / 2 - textMetrics.width / 2 - 3, this.plot.y(d.end_time), textMetrics.width + 6, (textMetrics.fontBoundingBoxAscent + textMetrics.fontBoundingBoxDescent));
                    context.fillStyle = "gray";
                    context.fillText(`end time: ${d.end_time.toFixed(2)}`, d.x + d.width / 2, this.plot.y(d.end_time) + 2);
                }
            }
        }

        draw() {
            this.plot.clear();
            this.drawDetails();
            this.drawPlayhead();
            this.plot.drawAxes();
        }

        update(viewArea, filteredWords) {
            this.plot.y.domain(viewArea);
            const [x0, x1] = viewArea;
            this.filtered_words = filteredWords;
            this.filtered_utterances = this.utterances.filter(w => w.start_time < x1 && w.end_time > x0);
            this.filtered_markers = this.markers ? this.markers.filter(m => m.start_time < x1 && m.end_time > x0) : null;
            this.filtered_matches = this.matches.filter(m => m.start_time <= x1 && m.end_time > x0);

            this.draw();
        }
    }

    class SelectedDetailsView {
        constructor(container) {
            // this.element contains the tooltip and the expand button
            this.element = container.append("div").classed("pill tooltip selection-details", true);

            // this.container contains the pills. Can be wrapped or not wrapped with overflow: hidden
            this.container = this.element.append("div").classed("selection-details-container", true);

            this.container.append("div").text("Selected segment:").classed("pill no-border info-label", true);

            this.expandButton = this.element.append("div").classed("selected-utterance-expand", true).html(icons['caret-down']).on("click", () => {
               this.container.classed("expanded", !this.container.classed("expanded"));
                if (this.container.classed("expanded")) {
                    this.expandButton.html(icons['caret-up'])
                } else {
                    this.expandButton.html(icons['caret-down'])
                }
            });
            this.update(null);

            this.blacklist = [
                "source", "session_id", "utterance_index", "utterance_overlaps", 
                "overlap_width", "overlap_shift", "num_columns", "x", "width", "highlight"
            ];
            this.rename = { total: "# words" }
        }

        clear() {
            this.element.selectAll(".utterance-details").remove();
            this.expandButton.style("visibility", "hidden");
        }

        formatValue(element, key, value) {
            if (/^([a-zA-Z0-9_/.-]+\.(wav|flac)(::\[[\d.:]+])?)$/.test(value)) {
                // Audio path: Display audio player

                let trials = []
                let url = new URL(value, window.location.href);
                if (window.location.protocol == 'file:') {
                    if (value[0] === '/'){
                        trials.push(settings.audio_server + value);
                        trials.push("file:////" + value);
                    } else {
                        trials.push(settings.audio_server + url.pathname);
                        trials.push(settings.audio_server + value);  // usually fails, but audio_server could define a prefix
                        trials.push("file:////" + url.pathname);
                        trials.push("file:////" + value);  // Can this work?
                    }
                } else {  // http or https
                    // file:// is not allowed in the browser for http(s), so we can't use it
                    if (value[0] === '/'){
                        trials.push(settings.audio_server + value);
                        trials.push(url.href);
                    } else {
                        trials.push(settings.audio_server + value);  // usually fails, but audio_server could define a prefix
                        // trials.push(settings.audio_server + url.pathname);  // could this work?
                        trials.push(url.href);
                    }
                }

                console.log("Audio candidates", trials)

                let audio = element.append("audio")
                audio.classed("info-value", true)
                    .attr("controls", "true")
                    .attr("src", trials.shift())  // pop the first entry from trials
                    .text(value);

                if (settings.show_playhead) {
                    let playheadTimer;
                    audio.on('play', e => {
                        playheadTimer = setInterval(() => {
                            state.playbackTime = audio.node().currentTime + state.selectedSegment.start_time;
                            details_plot.updatePlayhead(state.playbackTime);
                        }, 50);
                    });
                    audio.on('pause', e => {
                        clearInterval(playheadTimer);
                        state.playbackTime = null;
                        details_plot.updatePlayhead(state.playbackTime);
                    });
                }

                let fallback_text_box = element.append('div').classed("info-value", true)

                // Display tooltip with file path
                let tooltip = addTooltip(element, false).classed("wrap-60 alignleft", true)

                tooltip.append("div").attr('align', 'left').text(value)
                let warning_field = tooltip.append("div").attr('align', 'left').style("display", "none").html(
                    "\nIssue while loading\n  " + settings.audio_server + value +
                    "\nWith access to the audio file (either via audio server from the menu, or directly from the filesystem), a player will appear.\n" +
                    "Options:\n" +
                    " - With <code>python -m meeteval.viz.file_server</code> you can start a process, that exposes normalized wav files on http://localhost:7777\n" +
                    " - A standalone HTML file has access to the filesystem and doesn't need a server, but it cannot normalize the audio.\n" +
                    " - In Jupyter Notebooks only a server can deliver audio files.\n" +
                    "Slices (e.g. audio.wav::[0.5:1.0]) require a server."
                );

                // On error,
                //  - Add hint to tooltip when the play button works
                //  - Try to access local file
                //  - If that also doesn't work, show the file name and an
                //    exclamation mark to indicate an issue (Tooltip contains hints).

                let on_error_fn = function() {
                    if (trials.length > 0){
                        // Pop the first element
                        let next = trials.shift();
                        audio.attr("src", next);
                        if (!next.startsWith(settings.audio_server))
                            warning_field.style("display", "block");
                    } else {
                        warning_field.style("display", "block")
                        audio.remove();
                        fallback_text_box.text(value + ' ');
                        fallback_text_box.append('div').style("display", "inline-block").html(icons['warning']);
                    }
                };
                audio.on('error', on_error_fn)

                // Add a copy button
                let copy_button = element.append('button').classed("copybutton", true)
                let icon = copy_button.append("div");
                icon.html(icons['copy']);
                copy_button.on('click', function() {
                  // Click animation
                  navigator.clipboard.writeText(value);

                  icon.html(icons['check']);
                  setTimeout(function() {
                icon.html(icons['copy']);
                  }, 700)
                });


            } else {
                // Plain value: Display as text
                element.text(value);
            }
        }

        update(utterance) {
            this.clear();
            if (utterance) {
                this.expandButton.style("visibility", "visible");
                const tooltip = addTooltip(this.element).classed("wrap-60 alignleft utterance-details", true);
                const tooltipTable = tooltip.append("table").classed("details-table", true).append("tbody");

                for (var [key, value] of Object.entries(utterance)) {
                    if (this.blacklist.includes(key)) continue;
                    key = this.rename[key] || key;

                    // Pill
                    const pill = this.container.append("div").classed("utterance-details pill no-border", true);
                    pill.append("div").classed("info-label", true).text(key + ":");
                    this.formatValue(pill.append("div").classed("info-value", true), key, value);

                    // Row in tooltip table
                    const row = tooltipTable.append("tr").classed("utterance-details", true);
                    row.append("td").text(key + ":");
                    this.formatValue(row.append("td"), key, value);
                }
            } else {
                this.container.append("div").classed("utterance-details", true)
                    .classed("utterance-details-help pill no-border", true)
                    .text("Select a segment to display details");
            }
        }
    }

    class RangeSelector {
        constructor(container, state) {
            this.container = container.append("div").classed("range-selector", true).classed("pill", true);
            this.container.append("div").classed("info-label", true).text("Selection:");
            this.state = state;

            this.input = this.container.append("input")
                .attr("type", "text")
                .classed("range-selector-input", true)
                .attr("placeholder", "e.g. 0.0 - 1.0")
                // .attr("value", "0.0 - 1.0")
                .on("change", this._onChange.bind(this))
                .on("input", this._onInput.bind(this))
                .on("keydown", (event) => {
                    if (event.key === "Enter") {
                        this._onChange();
                    }
                });

            this.input.property(
                "value",
                `${state.viewAreas[state.viewAreas.length - 1][0]}-${state.viewAreas[state.viewAreas.length - 1][0]}`
            );

            this.parsedValue = [];
        }

        _onChange() {
            animateToViewArea(this.state.viewAreas.length - 1, this.parsedValue)
        }

        _onInput() {
            const value = this.input.node().value;

            let selection = parseSelection(value)

            if (selection) {
                const min_max = state.viewAreas[0];
                // Fix rounding issue of selection field. e.g. max is 360.78, while selection is 360.8
                let margin = 0.5

                if (selection[0] < min_max[0]-margin && selection[1] > min_max[1]+margin) {
                    this.input.classed("input-error", true);
                    selection = min_max;
                    console.log("Invalid range: outside of min and max", selection[0], selection[1], min_max, 'use', selection);
                } else if (selection[0] < min_max[0]-margin) {
                    this.input.classed("input-error", true);
                    selection = [min_max[0], Math.max(selection[1], min_max[1])];
                    console.log("Invalid range: below min", selection[0], selection[1], min_max, 'use', selection);
                } else if (selection[1] > min_max[1]+margin) {
                    this.input.classed("input-error", true);
                    selection = [Math.min(selection[0], min_max[0]), min_max[1]];
                    console.log("Invalid range: above max", selection[0], selection[1], min_max, selection);
                } else {
                    // This is the only valid case
                    this.input.classed("input-error", false);
                }
                this.parsedValue = selection;
                return;
            }

            // Hint that something is wrong
            this.input.classed("input-error", true);
        }

        update(viewArea) {
            viewArea = [viewArea[0].toFixed(1), viewArea[1].toFixed(1)]
            this.input.node().value = `${viewArea[0]} - ${viewArea[1]}`;
            this.input.classed("input-error", false);
        }
    }

    function drawRecordingAudioButton(container, rangeSelector, name, recording_file) {
        let update_audio = function (){
            // Note: Deleting and adding an audio with the same content doesn't
            //       trigger a load. Hence, no optimization necessary.
            audio_div.selectAll("*").remove();
            let lower = state.viewAreas[state.viewAreas.length - 1][0];
            let upper = state.viewAreas[state.viewAreas.length - 1][1];
            let range = lower + " - " + upper;
            let path = settings.audio_server + "/" + file_path.node().value + "?start=" + lower + "&stop=" + upper;
            if ( parseFloat(lower) < parseFloat(upper) ){
                audio_div.append("div").text(range);
                audio_div.append("audio")
                    .on("error", () => {audio_div.append("div").text("Issue while loading the audio")})
                    .attr("controls", "true")
                    .attr("src", path).text(path);
            } else {
                audio_div.append("div").text("Invalid range: " + range);
            }
        };
        let maybe_remove_audio = function (){
            if (state.viewAreas[state.viewAreas.length - 1]){
                let lower = state.viewAreas[state.viewAreas.length - 1][0];
                let upper = state.viewAreas[state.viewAreas.length - 1][1];
                let path = settings.audio_server + "/" + file_path.node().value + "?start=" + lower + "&stop=" + upper;
                let audio = audio_div.select("audio")
                if ( ! audio.empty() ){
                    if ( audio.attr("src") !== path ){
                        audio_div.selectAll("*").remove();
                    }
                    console.log(audio.attr("src") +  " vs " + path);
                }
            }
            if (settings.audio_server !== server_path.text())
                server_path.text(settings.audio_server);
        };

        const pill = container.append("div").classed("pill", true);
        const audio_tooltip = addTooltip(pill, null, maybe_remove_audio);

        pill.append("div").html(icons['audio']);
        if (name.trim()) pill.append("span").text(name).style("margin-left", "5px");

        const input = audio_tooltip.append("div").style("display", "flex");
        const server_path = input.append("div").text(settings.audio_server).attr("title", "To change the audio server address, edit the settings in the top left corner.");
        const file_path = input.append("input").attr("type", "text").style("width", "30em").attr("placeholder", "e.g. /path/to/file.wav").property("value", recording_file);

        const audio_div = audio_tooltip.append("div").style("display", "flex");

        input.append("button").text("Load").on("click", () => update_audio());
        file_path.on("keydown", (event) => {if (event.key === "Enter") {update_audio();}});
        pill.on("click", () => {update_audio(); audio_div.select("audio").node().play()});
    }

    // Setup plot elements
    const top_row_container = root_element.append("div").classed("top-row", true)
    drawHelpButton(top_row_container);
    drawMenu(top_row_container);
    drawExampleInfo(top_row_container, data.info)
    drawLegend(top_row_container);
    const rangeSelector = new RangeSelector(top_row_container, state);
    const searchBar = new SearchBar(top_row_container, state, settings.search_bar.initial_query);
    var minimaps = [];
    const selectedUtteranceDetails = new SelectedDetailsView(root_element.append("div").classed("top-row", true));
    // const status = d3.select(element_id).append("div").classed("top-row", true).append("div").text("status");

    for (const [key, value] of Object.entries(settings.recording_file)) {
        drawRecordingAudioButton(top_row_container, rangeSelector, key, value);
    }

    root_element.style("display", "flex").style("flex-direction", "column");
    const plot_div = root_element.append('div').classed("plot-area", true);

    var details_plot = null;
    function rebuild() {
        plot_div.selectAll("*").remove();
        minimaps = [];
        initializeViewAreas(time_domain);
        details_plot = null;

        for (let i = 0; i < settings.minimaps.number; i++) {
            minimaps.push(new Minimap(plot_div.append('div'), state, i));
        }

        details_plot = new DetailsPlot(
            new CanvasPlot(plot_div.append('div').style('flex-grow', '1'),
                d3.scaleBand().domain(speakers).padding(0.1),
                d3.scaleLinear().domain(time_domain),
                new DetailsAxis(30), new Axis(50), true, 2
            ), state, data.utterances, data.markers,
        )

        update();
    }

    rebuild();
    searchBar.search_button.node().click();

    function moveBy(offset) {
        animateToViewArea(state.viewAreas.length - 1, [state.viewAreas[state.viewAreas.length - 1][0] + offset, state.viewAreas[state.viewAreas.length - 1][1] + offset]);
    }

    // Register keyboard handler
    document.addEventListener("keydown", (event) => {
        if (!event.ctrlKey) {
            switch (event.key) {
                case "Escape": selectSegment(null); break;
                // Scroll by 10% of the currently visible range for ArrowUp and ArrowDown. A constant quickly feels too fast or too slow depending on the zoom level.
                case "ArrowUp": moveBy((state.viewAreas[state.viewAreas.length - 1][0] - state.viewAreas[state.viewAreas.length - 1][1]) / 10); break;
                case "ArrowDown": moveBy(-(state.viewAreas[state.viewAreas.length - 1][0] - state.viewAreas[state.viewAreas.length - 1][1]) / 10); break;
                // Scroll by the currently visible range for PageUP and PageDown
                case "PageUp": moveBy(state.viewAreas[state.viewAreas.length - 1][0] - state.viewAreas[state.viewAreas.length - 1][1]); break;
                case "PageDown": moveBy(state.viewAreas[state.viewAreas.length - 1][1] - state.viewAreas[state.viewAreas.length - 1][0]); break;
            }
        } else {
            switch(event.key) {
                // Go to previous/next error on Ctrl+ArrowUp/Down
                // This is useful when the recognition only contains a few errors and the user wants to quickly jump between them
                case "ArrowUp": selectNextMatchingSegment(u => u.errors > 0, true, true); break;
                case "ArrowDown": selectNextMatchingSegment(u => u.errors > 0, true, false); break;
            }
        } 
    });
}
