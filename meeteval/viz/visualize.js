var colormaps = {
    default: {
        'correct': 'lightgray',
        'substitution': '#F5B14D',  // yellow / orange
        'insertion': '#33c2f5', // blue
        'deletion': '#f2beb1',  // red
        // 'ignored': 'transparent',   // purple
        'highlight': 'green'
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
            height: 200,
        },
        show_details: true,
        show_legend: true,
        font_size: 12,
        search_bar: {
            initial_query: null
        },
        recording_file: "",
    }
) {
    // Validate settings
    for (const label of ['correct', 'substitution', 'insertion', 'deletion']) {
        if (settings.colors[label] === undefined) throw `Missing key in "colors" setting: ${label}`;
    }

    function call_throttled(fn, delay=5) {
        if (!fn.timerId) {
            // Call immediately
            fn()

            // Set timer to prevent further calls
            fn.timerId = setTimeout(() => {
                fn.timerId = undefined;
                fn.call_pending = false;
                if (fn.call_pending) fn();
            }, delay);
        } else {fn.call_pending = true;}
    }

    let root_element = d3.select(element_id);

class Axis {
    constructor(padding, numTicks=null, tickPadding=3, tickSize=6) {
        this.padding = padding;
        this.horizontal = true;
        this.tickPadding = tickPadding;
        this.tickSize = tickSize;
        this.numTicks = numTicks;
    }
    draw(
        context,
        scale,
        position,
    ) {
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
        context.font = "12px Arial";
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
    constructor(padding, label=null) {
        this.padding = padding;
        this.label = label;
    }

    draw(
        context,
        scale,
        position,
    ) {

        // Tick labels
        context.textAlign = "center";
        context.textBaseline = "top";

        const [start, end] = scale.range(),
            tickFormat = scale.tickFormat ? scale.tickFormat() : d => d,
            ticks = (scale.ticks ? scale.ticks() : scale.domain()).map(d => {
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
        context.font = "12px Arial";
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
    constructor(padding, tickPadding=3, tickSize=6, ref_hyp_gap=10) {
        this.padding = padding;
        this.tickPadding = tickPadding;
        this.tickSize = tickSize;
        this.ref_hyp_gap = ref_hyp_gap;
    }

    draw(context, scale, position) {
        const [start, end] = scale.range()
        const ticks = scale.domain().map(d => {
            return {
                pos: scale(d) + (scale.bandwidth ? scale.bandwidth() / 2 : 0),
                label: d
            }
        })
        const offset = (scale.bandwidth()) / 4;

        // Clear the axis part of the plot
        context.clearRect(start, position, end - start, this.padding);

        // Set up context
        context.lineWidth = 1;
        context.font = "12px Arial";
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
            context.moveTo(d.pos - this.ref_hyp_gap, position);
            context.lineTo(d.pos - this.ref_hyp_gap, position + this.tickSize);
            context.moveTo(d.pos + this.ref_hyp_gap, position);
            context.lineTo(d.pos + this.ref_hyp_gap, position + this.tickSize);
            context.moveTo(d.pos + scale.bandwidth() / 2, position);
            context.lineTo(d.pos + scale.bandwidth() / 2, position + this.tickSize);
            context.moveTo(d.pos - scale.bandwidth() / 2, position);
            context.lineTo(d.pos - scale.bandwidth() / 2, position + this.tickSize);
            context.stroke();
            context.fillStyle = "black";
            context.fillText(d.label, d.pos, position + this.tickSize + this.tickPadding)
            context.fillText("REF", d.pos - offset, position + this.tickPadding);
            context.fillText("HYP", d.pos + offset, position + this.tickPadding);
        });
    }
}

function addTooltip(element, tooltip) {
    element.classed("tooltip", true);
    const tooltipcontent = element.append("div").classed("tooltipcontent", true);
    if (typeof tooltip === "string") tooltipcontent.text(tooltip)
    else if (tooltip) tooltip(tooltipcontent);
    element.on("mouseenter", () => {
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
    });
    element.on("mouseleave", () => {
        tooltipcontent.node().style.translate = null;
    });
    return tooltipcontent;
}

class CanvasPlot {
    element;
    canvas;
    context;
    position;
    width;
    height;
    x_axis_padding;
    y_axis_padding;
    x;
    y;

    /**
     * Creates a canvas and axis elements to be drawn on a canvas plot
     *
     * @param element
     * @param width
     * @param height
     * @param x_scale
     * @param y_scale
     * @returns {{canvas, drawAxes: drawAxes, context: *, width, x: *, clear: clear, y: *, position: {x: number, y: number}, x_axis_padding: *, y_axis_padding: *, height}}
     */
    constructor(element, width, height, x_scale, y_scale, xAxis, yAxis, invert_y=false, x_axis_label='',) {
        this.element = element.append("div").style("position", "relative").style("height", height + "px");
        this.canvas = this.element.append("canvas").style("width", "100%").style("height", "100%");
        this.context = this.canvas.node().getContext("2d")
        this.width = width
        this.height = height
        this.xAxis = xAxis;
        this.yAxis = yAxis;
        this.x_axis_padding = xAxis?.padding || 0;
        this.y_axis_padding = yAxis?.padding || 0;
        this.invert_y = invert_y
        this.x_axis_label = x_axis_label;

        if (this.xAxis) this.xAxis.horizontal = true;
        if (this.yAxis) this.yAxis.horizontal = false;

        // Create plot elements
        this.x = x_scale;
        this.y = y_scale;
        this.sizeChangedListeners = [];
        this.canvasSizeChanged();

        // Track size changes of our canvas
        new ResizeObserver(this.canvasSizeChanged.bind(this)).observe(this.canvas.node());
    }

    onSizeChanged(callback) {
        this.sizeChangedListeners.push(callback);
    }

    canvasSizeChanged() {
        this.width = this.canvas.node().offsetWidth;
        this.height = this.canvas.node().offsetHeight;
        this.canvas.attr("width", this.width);
        this.canvas.attr("height", this.height);
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
        // Font
        const font_size = container.append("div").classed("pill", true)
        font_size.append("div").classed("info-label", true).text("Font size");
        font_size.append("input").attr("type", "range").attr("min", "5").attr("max", "30").classed("slider", true).attr("step", 1).on("input", function () {
            settings.font_size = this.value;
            redraw();
        }).node().value = settings.font_size;

        // Minimaps
        const minimaps = container.append("div").classed("pill", true);
        minimaps.append("div").classed("info-label", true).text("Minimaps");
        minimaps.append("div").text("#").classed("label", true);
        const num_minimaps_select = minimaps.append("select").on("change", function () {
            settings.minimaps.number = this.value;
            rebuild();
            redraw();
        });
        num_minimaps_select.append("option").attr("value", 0).text("0");
        num_minimaps_select.append("option").attr("value", 1).text("1");
        num_minimaps_select.append("option").attr("value", 2).text("2");
        num_minimaps_select.append("option").attr("value", 3).text("3");
        num_minimaps_select.node().value = settings.minimaps.number;

        minimaps.append("div").text("Error distribution").classed("label", true);
        // const errorbar_style = container.append("div").classed("pill", true);
        // errorbar_style.append("div").classed("info-label", true).text("Error distribution");
        const errorbar_style_select = minimaps.append("select").on("change", function () {
            settings.barplot.style = this.value;
            rebuild();
            redraw();
        });
        errorbar_style_select.append("option").attr("value", "absolute").text("Absolute");
        errorbar_style_select.append("option").attr("value", "relative").text("Relative");
        errorbar_style_select.append("option").attr("value", "hidden").text("Hidden");
        errorbar_style_select.node().value = settings.barplot.style;

        
        // const errorbar_mode = container.append("div").classed("pill", true);
        minimaps.append("div").text("Scale exclude correct").classed("label", true);
        // errorbar_mode.append("div").classed("info-label", true).text("Scale exclude correct");
        const errorbar_mode_check = minimaps.append("input").attr("type", "checkbox").on("change", function () {
            settings.barplot.scaleExcludeCorrect = this.checked;
            redraw();
        });
        errorbar_mode_check.node().checked = settings.barplot.scaleExcludeCorrect;
    }

    function drawHelpButton(container) {
        const pill = container.append("a").attr("href", "https://github.com/fgnt/meeteval").classed("pill", true)
        pill.append("i").classed("fas fa-question-circle", true);
        pill.append("div").text("Help");
    }

    function drawExampleInfo(container, info) {
        const root = container;

        label = (label, value, icon=null, tooltip=null) => {
            var l = root.append("div").classed("pill", true)
            if (icon) l.append("i").classed("fas " + icon, true);
            l.append("div").classed("info-label", true).text(label);
            l.append("div").classed("info-value", true).text(value);
            if (tooltip) addTooltip(l, tooltip);
            return l;
        }

        label("ID:", info.filename);
        label("Length:", info.length.toFixed(2) + "s");
        label("WER:", (info.wer.hypothesis.error_rate * 100).toFixed(2) + "%", null, c => {
            if (Object.keys(info.wer).length == 1){
                const wer = info.wer.hypothesis;
                const wer_by_speakers = info.wer_by_speakers.hypothesis;
                const table = c.append("table").classed("wer-table", true);
                const head = table.append("thead")
                const hr1 = head.append("tr");
                hr1.append("th");
                hr1.append("th");
                hr1.append("th");
                hr1.append("th");
                hr1.append("th").text("Counts by Speaker").attr("colspan", Object.keys(wer_by_speakers).length).style("border-bottom", "1px solid white");

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
            }
        });
        label("Alignment:", info.alignment_type, null,
            c => c.append('div').classed('wrap-40', true).text("The alignment algorithm used to generate this visualization. Available are:\n" +
            "cp: concatenated minimum-permutation\n" +
            "tcp: time-constrained minimum permutation\n\n" +
            "(This setting cannot be changed interactively, but has to be selected when generating the visualization)\n" + 
            "Check the documentation for details")
        )
        if (info.wer.reference_self_overlap?.overlap_rate) label(
            "Reference self-overlap:", 
            (info.wer.reference_self_overlap.overlap_rate * 100).toFixed(2) + "%", 
            "fa-triangle-exclamation",
            c => c.append('div').classed('wrap-40').text("Self-overlap is the percentage of time that a speaker annotation overlaps with itself. " +
            "On the reference, this is usually an indication for annotation errors.\n" +
            "Extreme self-overlap can lead to unexpected WERs!")
        ).classed("warn", true);
        if (info.wer.hypothesis_self_overlap?.overlap_rate) label(
            "Hypothesis self-overlap:", 
            (info.wer.hypothesis_self_overlap.overlap_rate * 100).toFixed(2) + "%",
            "fa-triangle-exclamation",
            c => c.append('div').classed('wrap-40').text("Self-overlap is the percentage of time that a speaker annotation overlaps with itself. " +
            "Extreme self-overlap can lead to unexpected WERs!")
        ).classed("warn", true);
    }


    /**
     * Search bar component. Modifies "words" in-place.
     */
    class SearchBar {
        constructor(container, words, initial_query) {
            this.words = words;
            this.container = container.append("div").classed("pill", true).classed("search-bar", true);
            this.text_input = this.container.append("input").attr("type", "text").attr("placeholder", "Regex (e.g., s?he)...");
            if (initial_query) this.text_input.node().value = initial_query;
            this.on_search_callbacks = [];

            // Start search when clicking on the button
            this.search_button = this.container.append("button").text("Search").on("click", () => this.search(this.text_input.node().value));

            // Start search on Ctrl + Enter
            this.text_input.on("keydown", (event) => {
                if (event.key === "Enter") {
                    this.search(this.text_input.node().value);
                }
            });
        }

        search(regex) {
            // Test all words against the regex. Use ^ and $ to get full match
            if (regex === "")  {
                this.words.forEach(w => w.highlight = false);
            } else {
                const re = new RegExp("^" + regex + "$", "i");
                for (const w of this.words) w.highlight = re.test(w.words);
            }
            this.on_search_callbacks.forEach(c => c());
        }

        onSearch(callback) {
            this.on_search_callbacks.push(callback);
        }
    }

    class ErrorBarPlot {
        constructor(canvas_plot, num_bins, words, style='absolute', scaleExcludeCorrect=false) {
            this.plot = canvas_plot;
            this.bin = d3.bin().thresholds(200).value(d => (d.start_time + d.end_time) / 2)
            this.words = words;
            this.max = 0;
            this.binned_words = [];
            this.style = style;
            this.scaleExcludeCorrect = scaleExcludeCorrect;
            this.plot.onSizeChanged(this.draw.bind(this));
            this.updateBins();
        }

        updateBins() {
            var bin_max = 0;
            const self = this;

            this.binned_words = this.bin.domain(this.plot.x.domain())(this.words).map(d => {
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

        zoomTo(x0, x1) {
            if ([x0, x1] == this.plot.x.domain()) return;
            this.plot.x.domain([x0, x1]);
            this.updateBins();
            this.draw();
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
        constructor(plot, words) {
            this.plot = plot;
            this.words = words;
            this.plot.onSizeChanged(this.draw.bind(this));
        }

        drawWords() {
            const [begin, end] = this.plot.x.domain();
            this.plot.context.strokeStyle = "gray";
            const bandwidth = this.plot.y.bandwidth() / 2;
            this.words.filter(d => d.start_time < end && d.end_time > begin).forEach(w => {
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

        zoomTo(x0, x1) {
            this.plot.x.domain([x0, x1]);
            this.draw();
        }

        draw() {
            this.plot.clear();
            this.drawWords();
            this.plot.drawAxes();
        }
    }

    class Minimap {
        constructor(element, width, height, x_scale, y_scale, words) {
            const e = element.append('div').classed("minimap", true)

            if (settings.barplot.style !== "hidden") {
                this.error_bars = new ErrorBarPlot(
                    new CanvasPlot(e, width, 40, x_scale,
                    d3.scaleLinear().domain([1, 0]),
                        null, new Axis(50, 3),
                ), 200, words, settings.barplot.style, settings.barplot.scaleExcludeCorrect);
            }
            this.word_plot = new WordPlot(
                new CanvasPlot(e, width, 100, x_scale, y_scale,
                    new CompactAxis(10, "time"), new Axis(50), true),
                words
            );

            if (settings.barplot.style !== "hidden") {
                this.error_bars.plot.element.append("div").classed("plot-label", true).style("margin-left", this.error_bars.plot.y_axis_padding + "px").text("Error distribution");
            }

            
            this.word_plot.plot.element.append("div").classed("plot-label", true).style("margin-left", this.word_plot.plot.y_axis_padding + "px").text("Segments");

            this.svg = e.append("svg")
                // .attr("width", width).attr("height", this.error_bars.plot.height + this.word_plot.plot.height)
                .style("position", "absolute").style("top", 0).style("left", 0).style("width", "100%").style("height", "100%");

            this.brush = d3.brushX()
                .extent([
                    [
                        Math.max(this.error_bars?.plot.y_axis_padding || 0, this.word_plot.plot.y_axis_padding),
                        0
                    ],
                    [this.word_plot.plot.width, this.word_plot.plot.height + (this.error_bars?.plot.height || 0)]])
                .on("brush", this._onselect.bind(this))
                .on("end", this._onselect.bind(this));

            this.brush_group = this.svg.append("g")
                .attr("class", "brush")
                .call(this.brush);

            this.on_select_callbacks = [];

            this.max_range = this.word_plot.plot.x.range();
            this.selection = this.word_plot.plot.x.range();

            // Redraw brush when size changes. This is required because the brush range / extent will otherwise keep the old value (in screen size)
            this.word_plot.plot.onSizeChanged(() => {
                this.brush.extent([
                    [Math.max(this.error_bars?.plot.y_axis_padding || 0, this.word_plot.plot.y_axis_padding), 0],
                    [this.word_plot.plot.width, this.word_plot.plot.height + (this.error_bars?.plot.height || 0)]]);
                this.brush_group.call(this.brush);
                // No idea how to keep the selection when the size changes, so we just keep the screen position
            });
        }

        draw() {
            if (this.error_bars) this.error_bars.draw();
            this.word_plot.draw();
        }

        zoomTo(x0, x1) {
            if (this.error_bars) this.error_bars.zoomTo(x0, x1);
            this.word_plot.zoomTo(x0, x1);
            this._callOnSelectCallbacks();
        }
        
        moveBrush(x0, x1) {
            this.brush_group.call(this.brush.move, [
                this.word_plot.plot.x(x0), this.word_plot.plot.x(x1)
            ])
        }

        removeBrush(){
            this.brush_group.call(this.brush.move, null);
        }

        _onselect(event) {
            if (event.selection === null) {
                if (this.selection[0] > this.max_range[0] && this.selection[1] < this.max_range[1]) this.selection = this.max_range;
            } else {
                this.selection = event.selection;
                // Remove brush when fully zoomed out
                if (this.selection[0] <= this.max_range[0] && this.selection[1] >= this.max_range[1]) {
                    this.removeBrush();
                }
            }
            this._callOnSelectCallbacks();
        }

        _callOnSelectCallbacks() {
            let [x0, x1] = this.selection;
            x0 = this.word_plot.plot.x.invert(x0);
            x1 = this.word_plot.plot.x.invert(x1);
            this.on_select_callbacks.forEach(c => c(x0, x1));
        }

        onSelect(callback) {
            this.on_select_callbacks.push(callback);
        }
    }

    class PlayHead {
        constructor(canvas, context, global_context, global_x_position, x_scale, y_scale, begin, end, src) {
            this.canvas = canvas;
            this.context = context;
            this.x = x_scale;
            this.y = y_scale;
            this.begin = begin;
            this.end = end;
            
            this.position = 0;
            this.animationFrameID = null;
            this.audio_data = null
            this.global_context = global_context
            this.global_x_position = global_x_position

            const self = this;
            if (src.includes('::')) {
                // TODO: support full spec
                
                const parts = src.split('::');
                const [begin, end] = parts[1].split(':');
                src = parts[0]
            } else {
                const [begin, end] = [0, null];
            }
            fetch(src)
                .then(r => r.arrayBuffer())
                .then(b => new AudioContext().decodeAudioData(b))
                .then(a => {
                    self.audio_data = a.getChannelData(0);
                    self.drawAudio();
                });
                
                
            this.h = new Howl({src: src})
            this.h.once('end', () => this.remove.bind(this))
            this.play();

        }

        clearPlayHead() {
            this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }

        draw() {
            const position = this.begin + (this.end - this.begin) * this.position;
            this.clearPlayHead();
            this.context.strokeStyle = 'pink';
            this.context.lineWidth = 4;
            this.context.beginPath();
            this.context.moveTo(this.x.bandwidth() / 2 + 20, this.y(position));
            this.context.lineTo(this.x.bandwidth(), this.y(position));
            this.context.stroke();
            this.context.beginPath();
            this.context.strokeStyle = 'black';
            this.context.lineWidth = 1;
            this.context.moveTo(this.x.bandwidth() / 2 + 20, this.y(position));
            this.context.lineTo(this.x.bandwidth(), this.y(position));
            this.context.stroke();
        }

        play() {
            this.h.stop();
            this.h.play();
            this.tick();
        }

        stop() {
            this.h.stop();
        }

        tick() {
            this.position = (this.h.seek() || 0) / this.h.duration();
            this.draw();
            this.animationFrameID = requestAnimationFrame(this.tick.bind(this));
        }

        remove() {
            if (this.animationFrameID !== null) cancelAnimationFrame(this.animationFrameID);
            this.clearPlayHead();
        }

        drawAudio() {
            if (this.audio_data == null) return;
            const begin = this.y(this.begin)
            const end = this.y(this.end)
            const length = end - begin;
            const data_length = this.audio_data.length;
            const scale = length / data_length;
            var prevY = 0;
            var max = 0;
            const vscale = 100

            this.global_context.beginPath()
            this.global_context.moveTo(this.global_x_position, begin);
            for (let i = 0; i <= data_length; i++) {
                const y = Math.round(i * scale)
                if (y > prevY) {
                    const x = this.global_x_position + Math.round(max * vscale)
                    this.global_context.lineTo(x, prevY + begin)
                    prevY = y;
                    max = 0;
                }
                max = Math.max(Math.abs(this.audio_data[i] || 0), max)
            }
            this.global_context.fillStyle = 'gray';
            this.global_context.fill();
        }
    }


    class DetailsPlot {
        constructor(plot, words, utterances, markers, ref_hyp_gap=10) {
            this.plot = plot;
            this.plot.element.classed("minimap", true)
            this.words = words;
            this.filtered_words = words;
            this.utterances = utterances;
            this.filtered_utterances = utterances;
            this.max_length = plot.y.domain()[1];
            this.ref_hyp_gap = ref_hyp_gap;
            this.markers = markers;
            this.filtered_markers = markers;

            // Precompute alignment / stitches for insertions/substitutions
            this.matches = words.flatMap((w) => {
                if (!w.matches) return [];
                return w.matches
                    .filter(m => m[0])  // Only look at matches between words
                    .map(m => {
                        const other = words[m[0]];
                        const [left, right] = w.source === "hypothesis" ? [other, w] : [w, other];
                        return {
                            speaker: w.speaker,
                            match_type: m[1],
                            left_center_time: left.center_time,
                            right_center_time: right.center_time,
                            start_time: Math.min(left.center_time, right.center_time),
                            end_time: Math.max(left.center_time, right.center_time),
                        }
                    })
            });
            this.filtered_matches = this.matches;

            this.selected_utterance = null;
            this.playhead = null;
            this.utteranceSelectListeners = [];

            // Playhead canvas on top of the plot canvas
            this.playhead_canvas = this.plot.element.append("canvas").style("position", "absolute").style("top", 0).style("left", 0).style("width", "100%").style("height", "100%");

            this.onUtteranceSelect(this.draw.bind(this));
            // this.onUtteranceSelect(this.play.bind(this));

            // Plot label
            this.plot.element.append("div").classed("plot-label", true).style("margin-left", this.plot.y_axis_padding + "px").text("Detailed matching");

            const self = this;
            this.plot.element.on("click", (event) => {
                const screenX = event.layerX;
                const screenY = event.layerY;
                const y = self.plot.y.invert(screenY);

                // invert x band scale
                const eachBand = self.plot.x.step();
                const index = Math.floor((screenX - self.plot.y_axis_padding) / eachBand);
                const speaker = self.plot.x.domain()[index];
                const within_speaker_coord = screenX - self.plot.x(speaker);
                const source = within_speaker_coord < self.plot.x.bandwidth() / 2 - self.ref_hyp_gap ? "reference" : (within_speaker_coord > self.plot.x.bandwidth() / 2 + self.ref_hyp_gap ? "hypothesis" : null);

                if (source) {
                    const utterance_candidates = this.filtered_utterances.filter(
                        u => u.start_time < y && u.end_time > y && u.speaker === speaker && u.source === source
                    )
                    if (utterance_candidates.length > 0) this.selectUtterance(utterance_candidates[0]);
                    else this.selectUtterance(null);
                } else this.selectUtterance(null);
            })

            this.plot.element.on("wheel", (event) => {
                let [begin, end] = this.plot.y.domain();
                let delta = (this.plot.y.invert(event.deltaY) - this.plot.y.invert(0)) * 0.5    // TODO: magic number
                if (event.ctrlKey) {
                    // Zoom when ctrl is pressed. Zoom centered on mouse position
                    const mouse_y = this.plot.y.invert(event.layerY);
                    const ratio = (mouse_y - begin) / (end - begin);
                    begin = Math.max(0, begin - delta * ratio);
                    end = Math.min(end + delta * (1-ratio), this.max_length);
                } else {
                    // Move when ctrl is not pressed
                    if (end + delta > this.max_length) delta = this.max_length - end;
                    if (begin + delta < 0) delta = -begin;
                    begin = begin + delta;
                    end = end + delta;
                }
                // TODO: We shouldn't call zoomTo here because it would create an update loop
                this._callOnScrollHandlers(begin, end);
                event.preventDefault();
            }, false)

            this.plot.element.on("mousemove", event => {
                if (event.buttons !== 1) return;
                const delta = this.plot.y.invert(event.movementY) - this.plot.y.invert(0);
                let [begin, end] = this.plot.y.domain();
                this._callOnScrollHandlers(begin - delta, end - delta);
            })

            var lastTouchY = [];
            this.plot.element.on("touchstart", event => {
                // TouchList doesn't implement iterator
                alert("touchstart")
                lastTouchY = [];
                for (let i = 0; i < event.touches.length; i++) {
                    lastTouchY.push(event.touches[i].screenY);
                }
            });
            this.plot.element.on("touchend", event => {
                // TouchList doesn't implement iterator
                alert("touchend")
                lastTouchY = [];
                for (let i = 0; i < event.touches.length; i++) {
                    lastTouchY.push(event.touches[i].screenY);
                }
            });
            
            this.plot.element.on("touchmove", event => {
                // TODO: fling?
                // TouchList doesn't implement iterator
                var touchY = [];
                for (let i = 0; i < event.touches.length; i++) {
                    touchY.push(event.touches[i].screenY);
                }
                if (lastTouchY) {
                    // Use the delta between the touches that are furthest apart
                    const minY = Math.min(...touchY);
                    const maxY = Math.max(...touchY);
                    const lastMinY = Math.min(...lastTouchY);
                    const lastMaxY = Math.max(...lastTouchY);

                    // Move center to the center of the touch points
                    const center = this.plot.y.invert((maxY + minY) / 2);
                    const lastCenter = this.plot.y.invert((lastMaxY + lastMinY) / 2);
                    const delta = lastCenter - center;
                    let [begin, end] = this.plot.y.domain();
                    begin += delta;
                    end += delta;
                    
                    // Zoom so that the center point doesn't move 
                    // TODO: this computation is _slightly_ off, but I don't know why
                    if (lastMaxY - lastMinY > 0 && maxY - minY > 0) { 
                        const ratio = (maxY - minY) / (lastMaxY - lastMinY);
                        const zoomDelta = (end - begin) * (ratio - 1);
                        const positionRatio = (center - begin) / (end - begin);
                        begin = Math.max(0, begin + zoomDelta * positionRatio);
                        end = Math.min(end - zoomDelta * (1-positionRatio), this.max_length);
                    }
                    this._callOnScrollHandlers(begin, end);
                    event.preventDefault()
                }
                lastTouchY = touchY;
            })

            this.onscrollhandlers = [];

            this.plot.onSizeChanged(this.draw.bind(this));
        }

        onUtteranceSelect(callback) {
            this.utteranceSelectListeners.push(callback);
        }

        selectUtterance(utterance) {
            this.selected_utterance = utterance;
            this.utteranceSelectListeners.forEach(c => c(utterance));
        }

        play(utterance) {
            this.playhead = new PlayHead(
                this.playhead_canvas, this.playhead_canvas.node().getContext("2d"),
                this.plot.context, 0, this.plot.x, this.plot.y, 
                utterance.start_time,  utterance.end_time, utterance.audio
            )
        }

        onScroll(callback) {
            this.onscrollhandlers.push(callback);
        }

        _callOnScrollHandlers(x0, x1) {
            this.onscrollhandlers.forEach(c => c(x0, x1));
        }

        drawDetails() {
            const filtered_words = this.filtered_words;
            if (filtered_words.length > 3000) {
                this.plot.context.font = "30px Arial";
                this.plot.context.textAlign = "center";
                this.plot.context.textBaseline = "middle";
                this.plot.context.fillStyle = "gray";
                this.plot.context.fillText("Zoom in or select a smaller region in the minimap above", this.plot.width / 2, this.plot.height / 2);
                return;
            }
            const filtered_utterances = this.filtered_utterances;
            const context = this.plot.context;

            const draw_text = filtered_words.length < 400;
            const draw_boxes = filtered_words.length < 1000;
            const draw_utterance_markers = filtered_words.length < 2000;
            const rectwidth = this.plot.x.bandwidth() / 2 - this.ref_hyp_gap;
            const bandwidth = this.plot.x.bandwidth() / 2;

            // Draw background
            for (let i = 0; i < this.plot.x.domain().length; i++) {
                const speaker = this.plot.x.domain()[i];
                const y = this.plot.y.range()[0];
                const x = this.plot.x(speaker);
                const width = this.plot.x.bandwidth();
                const height = this.plot.y.range()[1] - this.plot.y.range()[0];
                context.fillStyle = "#eee";
                context.fillRect(x, y, width, height);
            }

            // Draw utterance begin/end markers
            if (draw_utterance_markers) {
                context.strokeStyle = "black";
                context.lineWidth = .1;
                // context.beginPath();
                // filtered_utterances.forEach(d => {
                    //     var y = this.plot.y(d.start_time) - 1;
                    //     context.moveTo(minX, y);
                    //     context.lineTo(maxX, y);
                    //     y = this.plot.y(d.end_time) + 1;
                    //     context.moveTo(minX, y);
                    //     context.lineTo(maxX, y);
                    // });
                    // context.stroke();
                    
                if (this.selected_utterance) {
                    const [minX, maxX] = this.plot.x.range();
                    context.lineWidth = .5;
                    context.strokeStyle = 'red';
                    var y = this.plot.y(this.selected_utterance.start_time) - 1;
                    context.beginPath();
                    context.moveTo(minX, y);
                    context.lineTo(maxX, y);
                    y = this.plot.y(this.selected_utterance.end_time) + 1;
                    context.moveTo(minX, y);
                    context.lineTo(maxX, y);
                    context.stroke();
                }
            }

            const filtered_markers = this.filtered_markers;
            // Draw a range marker on the left side of the plot with two lines spanning the full width
            if (filtered_markers) filtered_markers.forEach(m => {
                // TODO: custom color and label, markers for speakers
                if (m.type === "range") {
                    const y0 = this.plot.y(m.start_time);
                    const y1 = this.plot.y(m.end_time);
                    context.fillStyle = "purple";
                    context.strokeStyle = "purple";
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
                    context.fillStyle = "purple";
                    context.strokeStyle = "purple";
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
                const bandleft = this.plot.x(d.speaker);
                let rectleft = bandleft;
                if (d.source === "hypothesis") rectleft += bandwidth + this.ref_hyp_gap;

                if (d.matches?.length > 0 || d.highlight) {
                    context.beginPath();
                    context.rect(
                        rectleft,
                        this.plot.y(d.start_time),
                        rectwidth,
                        this.plot.y(d.end_time) - this.plot.y(d.start_time));
                    
                    if (d.highlight) context.fillStyle = settings.colors.highlight;
                    else context.fillStyle = settings.colors[d.matches[0][1]];
                }

                context.fill();
                context.strokeStyle = "gray";
                context.lineWidth = 2;
                if (draw_boxes) context.stroke();
                
                // Stitches for insertion / deletion
                if (d.matches?.length > 0) {
                    // TODO: support multiple matches
                    const [match_index, match_type] = d.matches[0];
                    context.beginPath();
                    context.lineWidth = 2;
                    context.strokeStyle = settings.colors[match_type];
                    if (match_type === 'insertion') {
                        const y = this.plot.y(d.center_time);
                        context.moveTo(rectleft, y);
                        context.lineTo(rectleft - this.ref_hyp_gap / 2, y);
                    } else if (match_type === 'deletion') {
                        const y = this.plot.y(d.center_time);
                        context.moveTo(rectleft + rectwidth, y);
                        context.lineTo(rectleft + rectwidth + this.ref_hyp_gap / 2, y);
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
                context.moveTo(bandleft + rectwidth, this.plot.y(m.left_center_time));
                context.lineTo(bandleft + rectwidth + this.ref_hyp_gap / 2, this.plot.y(m.left_center_time));
                context.lineTo(bandleft + rectwidth + 3 * this.ref_hyp_gap / 2, this.plot.y(m.right_center_time));
                context.lineTo(bandleft + rectwidth + 2 * this.ref_hyp_gap, this.plot.y(m.right_center_time));
                context.stroke();
            });

            // Draw word text
            context.font = `${settings.font_size}px Arial`;
            context.textAlign = "center";
            context.textBaseline = "middle";
            context.lineWidth = 1;

            if (draw_text) filtered_words.forEach(d => {
                const bandleft = this.plot.x(d.speaker);
                let rectleft = bandleft;
                if (d.source === "hypothesis") rectleft += bandwidth + this.ref_hyp_gap;

                rectleft += rectwidth / 2;
                let y_ = this.plot.y((d.start_time + d.end_time) / 2);
                if (d.matches === undefined) context.fillStyle = "gray";
                else context.fillStyle = '#000';
                context.fillText(d.words, rectleft, y_);
            })

            // Draw utterance begin and end markers
            const markerLength = 6;
            const markerOverhang = 3;
            if (draw_utterance_markers) filtered_utterances.forEach(d => {
                context.strokeStyle = "black";
                context.lineWidth = 1.5;
                context.beginPath();

                // x is the left side of the marker
                var x = this.plot.x(d.speaker);
                const bandwidth = this.plot.x.bandwidth() / 2 - this.ref_hyp_gap;
                if (d.source == "hypothesis") {
                    x += bandwidth + 2*this.ref_hyp_gap;
                }

                // Begin marker
                var y = this.plot.y(d.start_time) - 1;
                context.moveTo(x - markerOverhang, y + markerLength);
                context.lineTo(x - markerOverhang, y);
                context.lineTo(x + bandwidth + markerOverhang, y);
                context.lineTo(x + bandwidth + markerOverhang, y + markerLength);

                // End marker
                y = this.plot.y(d.end_time) + 1;
                context.moveTo(x - markerOverhang, y - markerLength);
                context.lineTo(x - markerOverhang, y);
                context.lineTo(x + bandwidth + markerOverhang, y);
                context.lineTo(x + bandwidth + markerOverhang, y - markerLength);
                context.stroke();

                // Draw marker that text is empty
                if (d.words === "" && draw_text) {
                    context.beginPath();
                    context.textAlign = "center";
                    context.textBaseline = "middle";
                    context.strokeStyle = "lightgray";
                    context.linewidth = 1;
                    const x_ = x + bandwidth / 2;
                    context.font = `italic ${settings.font_size}px Arial`;
                    context.fillStyle = "gray";
                    context.fillText('(empty segment)', x_, (this.plot.y(d.start_time) + this.plot.y(d.end_time)) / 2);
                }
            });

            // Draw boundary around the selected utterance
            if (this.selected_utterance) {
                const d = this.selected_utterance;
                const x = this.plot.x(d.speaker) + (d.source === "hypothesis" ? bandwidth + this.ref_hyp_gap : 0);
                context.beginPath();
                context.strokeStyle = "red";
                context.lineWidth = 3;
                context.rect(x, this.plot.y(d.start_time), rectwidth, this.plot.y(d.end_time) - this.plot.y(d.start_time));
                context.stroke();

                // Write begin time above begin marker
                context.font = `italic ${settings.font_size}px Arial`;
                context.fillStyle = "gray";
                context.textAlign = "center";
                context.textBaseline = "bottom";
                context.fillText(`begin time: ${d.start_time.toFixed(2)}`, x + rectwidth / 2, this.plot.y(d.start_time) - 3);

                // Write end time below end marker
                context.textBaseline = "top";
                context.fillText(`end time: ${d.end_time.toFixed(2)}`, x + rectwidth / 2, this.plot.y(d.end_time) + 3);
            }
        }

        draw() {
            this.plot.clear();
            this.drawDetails();
            if (this.playhead !== null) this.playhead.drawAudio();
            this.plot.drawAxes();
            // this.drawYAxisLabels();
        }

        zoomTo(x0, x1) {
            this.plot.y.domain([x0, x1]);
            this.filtered_words = this.words.filter(w => w.start_time < x1 && w.end_time > x0);
            this.filtered_utterances = this.utterances.filter(w => w.start_time < x1 && w.end_time > x0);
            this.filtered_markers = this.markers ? this.markers.filter(m => m.start_time < x1 && m.end_time > x0) : null;
            this.filtered_matches = this.matches.filter(m => m.start_time <= x1 && m.end_time > x0);

            call_throttled(this.draw.bind(this));
        }
    }

    class SelectedDetailsView {
        constructor(container) {
            this.container = container.append("div").classed("pill tooltip selection-details", true);
            this.container.append("div").text("Selected segment:").classed("pill no-border info-label", true);
            this.update(null);

            this.blacklist = ["source"]
            this.rename = { total: "# words" }
        }

        clear() {
            this.container.selectAll(".utterance-details").remove();
        }

        formatValue(element, key, value) {
            if (/^([a-zA-Z0-9_/.-]+\.wav)$/.test(value)) {
                // Audio path: Display audio player
                let audio = element.append("audio")
                audio.classed("info-value", true)
                    .attr("controls", "true")
                    .attr("src", "http://localhost:7777" + value)
                    .text(value);

                let fallback_text_box = element.append('div').classed("info-value", true)

                // Display tooltip with file path
                let tooltip = addTooltip(element, false).classed("wrap-60 alignleft", true)

                tooltip.append("div").attr('align', 'left').text(value)

                // On error,
                //  - Add hint to tooltip when the play button works
                //  - Try to access local file
                //  - If that also doesn't work, show the file name and an
                //    exclamation mark to indicate an issue (Tooltip contains hints).
                audio.on('error', function() {
                    audio.attr("src", "file:////" + value);
                    tooltip.append("div").attr('align', 'left').text(
                        "\nWith access to the audio file, a player will appear. Options:\n" +
                        " - With 'python -m meeteval.viz.file_server' you can start a process, that exposes normalized wav files on http://localhost:7777\n" +
                        " - A standalone HTML file has access to the filesystem and doesn't need a server, but it cannot normalize the audio.\n" +
                        " - In Jupyter Notebooks only a server can deliver audio files."
                    );
                    audio.on('error', function() {
                        audio.remove();
                        fallback_text_box.text(value + ' ');
                        fallback_text_box.append("i").classed("fas fa-triangle-exclamation", true);
                        copy_button;
                    })
                });

                // Add a copy button
                let copy_button = element.append('button').classed("copybutton", true)
                let icon = copy_button.append("i");
                icon.classed("fas fa-copy", true);
                copy_button.on('click', function() {
                  // Click animation
                  navigator.clipboard.writeText(value);

                  icon.classed("fas fa-copy", false);
                  icon.classed("fas fa-check", true);
                  setTimeout(function() {
                      icon.classed("fas fa-check", false);
                      icon.classed("fas fa-copy", true);
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
                const tooltip = addTooltip(this.container).classed("wrap-60 alignleft utterance-details", true);
                const tooltipTable = tooltip.append("table").classed("details-table", true).append("tbody");

                for (var [key, value] of Object.entries(utterance)) {
                    if (this.blacklist.includes(key)) return;
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
                this.container.append("div").classed("utterance-details", true).classed("utterance-details-help pill no-border", true).text("Select a segment to display details");
            }
        }
    }

    class RangeSelector {
        constructor(container) {
            this.container = container.append("div").classed("range-selector", true).classed("pill", true);
            this.container.append("div").classed("info-label", true).text("Selection:");
            this.lower_input = this.container.append("input").attr("type", "number").classed("range-selector-input", true).attr("min", 0).attr("max", 1).attr("step", 0.01).attr("value", 0).on("change", this._onSelect.bind(this))
            this.container.append("span").text("-")
            this.upper_input = this.container.append("input").attr("type", "number").classed("range-selector-input", true).attr("min", 0).attr("max", 1).attr("step", 0.01).attr("value", 0).on("change", this._onSelect.bind(this))
            this.on_select_callbacks = [];
        }

        _onSelect() {
            let [a, b] = [this.lower_input.node().value, this.upper_input.node().value];
            if (a < b) this.on_select_callbacks.forEach(c => c(a, b));
        }

        onSelect(callback) {
            this.on_select_callbacks.push(callback);
        }

        zoomTo(x0, x1) {
            this.lower_input.node().value = x0.toFixed(1);
            this.upper_input.node().value = x1.toFixed(1);
        }
    }

    function drawRecordingAudioButton(container, rangeSelector, name, recording_file) {
        let update_audio = function (){
            // Note: Deleting and adding an audio with the same content doesn't
            //       trigger a load. Hence, no optimization necessary.
            audio_div.selectAll("*").remove();
            let lower = rangeSelector.lower_input.node().value;
            let upper = rangeSelector.upper_input.node().value;
            let range = lower + " - " + upper;
            let path = server_path.node().value + "/" + file_path.node().value + "?start=" + lower + "&stop=" + upper;
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
            let lower = rangeSelector.lower_input.node().value;
            let upper = rangeSelector.upper_input.node().value;
            let path = server_path.node().value + "/" + file_path.node().value + "?start=" + lower + "&stop=" + upper;
            let audio = audio_div.select("audio")
            if ( ! audio.empty() ){
                if ( audio.attr("src") !== path ){
                    audio_div.selectAll("*").remove();
                }
                console.log(audio.attr("src") +  " vs " + path);
            }
        };

        let pill = container.append("div").classed("pill", true);
        let audio_tooltip = addTooltip(pill);

        pill.append("span").text((name.trim() !== "") ? "🔊 " + name : "🔊");

        let input = audio_tooltip.append("div").style("display", "flex");
        let server_path = input.append("input").attr("type", "text").attr("placeholder", "e.g. http://localhost:7777").property("value", "http://localhost:7777");
        let file_path = input.append("input").attr("type", "text").style("width", "30em").attr("placeholder", "e.g. /path/to/file.wav").property("value", recording_file);

        let audio_div = audio_tooltip.append("div").style("display", "flex");

        input.append("button").text("Load").on("click", () => update_audio());
        server_path.on("keydown", (event) => {if (event.key === "Enter") {update_audio();}});
        file_path.on("keydown", (event) => {if (event.key === "Enter") {update_audio();}});
        pill.on("click", () => {update_audio(); audio_div.select("audio").node().play()});

        pill.on("mouseenter", maybe_remove_audio);

        update_audio()
    }

    // Data preprocessing
    const time_domain = [0, Math.max.apply(null, (data.utterances.map(d => d.end_time))) + 1];
    const speakers = data.utterances .map(d => d.speaker)

    // Setup plot elements
    var margin = {top: 30, right: 30, bottom: 70, left: 60},
        width = 1500 - margin.left - margin.right,
        height = 300 - margin.top - margin.bottom;

    const top_row_container = root_element.append("div").classed("top-row", true)
    drawHelpButton(top_row_container);
    drawExampleInfo(top_row_container, data.info)
    // drawMenuBar(top_row_container);
    if (settings.show_legend) drawLegend(top_row_container);
    drawMenu(top_row_container);
    const rangeSelector = new RangeSelector(top_row_container);
    const searchBar = new SearchBar(top_row_container, data.words, settings.search_bar.initial_query);
    var minimaps = [];
    searchBar.onSearch(() => {
        for (const minimap of minimaps) minimap.error_bars.updateBins();
        redraw();
    });
    const selectedUtteranceDetails = new SelectedDetailsView(root_element.append("div").classed("top-row", true));
    // const status = d3.select(element_id).append("div").classed("top-row", true).append("div").text("status");

    for (const [key, value] of Object.entries(settings.recording_file)) {
        drawRecordingAudioButton(top_row_container, rangeSelector, key, value);
    }

    const plot_container = root_element.append("div").style("margin", "10px")
    const plot_div = plot_container.append("div").style("position", "relative")

    var details_plot = null;
    function rebuild() {
        plot_div.selectAll("*").remove();
        minimaps = [];
        details_plot = null;

        for (let i = 0; i < settings.minimaps.number; i++) {
            const minimap = new Minimap(
                plot_div, width, settings.minimaps.height,
                d3.scaleLinear().domain(time_domain),
                d3.scaleBand().domain(speakers).padding(0.1),
                data.words,
            )
            if (minimaps[i-1] !== undefined) {
                minimaps[i-1].onSelect(minimap.zoomTo.bind(minimap));
            }
            minimaps.push(minimap);
        }

        if (settings.show_details) {
            details_plot = new DetailsPlot(
                new CanvasPlot(plot_div, width, 700,
                    d3.scaleBand().domain(speakers).padding(0.1),
                    d3.scaleLinear().domain([time_domain[0], time_domain[1]]),
                    new DetailsAxis(30), new Axis(50), true
                ), data.words, data.utterances, data.markers
            )

            details_plot.onUtteranceSelect(selectedUtteranceDetails.update.bind(selectedUtteranceDetails));

            if (minimaps.length > 0) {
                const last_minimap = minimaps[minimaps.length - 1];
                last_minimap.onSelect(details_plot.zoomTo.bind(details_plot));
                last_minimap.onSelect(rangeSelector.zoomTo.bind(rangeSelector));
                details_plot.onScroll(last_minimap.moveBrush.bind(last_minimap));
                rangeSelector.onSelect(last_minimap.moveBrush.bind(last_minimap));
            } else {
                // This is necessary to prevent update loops. We can't call details_plot.zoomTo in details_plot...
                details_plot.onScroll(details_plot.zoomTo.bind(details_plot));
                details_plot.onScroll(rangeSelector.zoomTo.bind(rangeSelector));
                rangeSelector.onSelect(details_plot.zoomTo.bind(details_plot));
            }
        }
    }

    function redraw() {
        if (settings.show_details) details_plot.draw();
        for (const minimap of minimaps) minimap.draw();
    }

    rebuild();
    searchBar.search(settings.search_bar.initial_query);
    redraw();
}
