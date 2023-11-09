var colormaps = {
    default: {
        'correct': 'lightgray',
        'substitution': '#F5B14D',  // yellow / orange
        'insertion': '#3F6193', // blue
        'deletion': '#B3381A',  // red
    },
    diff: {
        'correct': 'lightgray',
        'substitution': 'yellow',
        'insertion': 'green',
        'deletion': 'red',
    },
    seaborn_muted: {
        'correct': 'lightgray',
        'substitution': '#dd8452',  // yellow
        'insertion': '#4c72b0', // blue
        'deletion': '#c44e52',  // red
    }
}

function alignment_visualization(
    data,
    element_id = '#my_dataviz',
    settings= {
        colors: colormaps.default,
        barplot: {
            style: 'absolute', // hidden, absolute, relative
            scaleExcludeCorrect: false
        },
        minimaps: {
            number: 2,
            height: 200,
        },
        show_details: true,
        show_legend: true,
        font_size: 12,
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

    /**
 * Draw a single axis, as defined by a d3 scale, on a canvas using its 2d context
 *
 * @param context The canvas' 2d context
 * @param scale The d3 scale that defines the axis / space
 * @param position The position of the axis from the bottom (horizontal) or left (vertical) of the canvas
 * @param padding The starting point of the axis line (padding of the other axis) from the left (horizontal) or bottom (vertical) of the canvas
 * @param horizontal Whether the axis is horizontal or vertical
 * @param tickPadding The padding between the tick marks and the tick labels
 * @param tickSize The size/length of the tick marks
 */
function drawAxis(
    context,
    scale,
    position,
    padding,
    horizontal,
    tickPadding = 3,
    tickSize = 6,
) {
    const [start, end] = scale.range(),
        tickFormat = scale.tickFormat ? scale.tickFormat() : d => d,
        ticks = (scale.ticks ? scale.ticks() : scale.domain()).map(d => {
            return {
                pos: scale(d) + (scale.bandwidth ? scale.bandwidth() / 2 : 0),
                label: tickFormat(d)
            }
        });

    // Flip coords if vertical
    let coord, c;
    if (horizontal) {
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
    context.clearRect(...coord(start, 0), ...c(end - start, padding));

    // Tick marks
    context.beginPath();
    ticks.forEach(d => {
        context.moveTo(...coord(d.pos, 0));
        context.lineTo(...coord(d.pos, tickSize));
    });
    // Line
    context.moveTo(...coord(start, tickSize));
    context.lineTo(...coord(start, 0));
    context.lineTo(...coord(end, 0));
    context.lineTo(...coord(end, tickSize));
    context.stroke();

    // Tick labels
    if (horizontal) {
        context.textAlign = "center";
        context.textBaseline = "top";
    } else {
        context.textAlign = "right";
        context.textBaseline = "middle";
    }

    ticks.forEach(d => {
        context.fillText(d.label, ...coord(d.pos, tickSize + tickPadding));
    });
}

  /**
 * Draw a single axis, as defined by a d3 scale, on a canvas using its 2d context
 *
 * @param context The canvas' 2d context
 * @param scale The d3 scale that defines the axis / space
 * @param position The position of the axis from the bottom (horizontal) or left (vertical) of the canvas
 * @param padding The starting point of the axis line (padding of the other axis) from the left (horizontal) or bottom (vertical) of the canvas
 * @param horizontal Whether the axis is horizontal or vertical
 * @param tickPadding The padding between the tick marks and the tick labels
 * @param tickSize The size/length of the tick marks
 */
function drawAxisCompact(
    context,
    scale,
    position,
    padding,
    horizontal,
    tickPadding = 3,
    tickSize = 6,
) {

    // Tick labels
    if (horizontal) {
        context.textAlign = "center";
        context.textBaseline = "top";
    } else {
        context.textAlign = "right";
        context.textBaseline = "middle";
    }

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
    if (horizontal) {
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
    context.clearRect(...coord(start, 0), ...c(end - start, padding));

    context.beginPath();
    // Line
    const p = (ticks[0].textMetrics.fontBoundingBoxAscent + ticks[0].textMetrics.fontBoundingBoxDescent) / 2;
    context.lineTo(...coord(start, p));
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
    constructor(element, width, height, x_scale, y_scale, invert_y=false, draw_x_axis=true, draw_y_axis=true, x_axis_label='',) {
        this.element = element.append("div").style("position", "relative").style("height", height + "px");
        this.canvas = this.element.append("canvas").style("width", "100%").style("height", "100%");
        this.context = this.canvas.node().getContext("2d")
        this.position = {x: 0, y: 0}
        this.width = width
        this.height = height
        this.x_axis_padding = draw_x_axis ? 10 : 0;
        this.y_axis_padding = draw_y_axis ? 50 : 0;
        this.invert_y = invert_y
        this.draw_x_axis = draw_x_axis;
        this.draw_y_axis = draw_y_axis;
        this.x_axis_label = x_axis_label;

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
        this.x.range([this.position.x + this.y_axis_padding, this.position.x + this.width])
        if (this.invert_y) {
            this.y.range([this.position.y, this.position.y + this.height - this.x_axis_padding])
        } else {
            this.y.range([this.position.y + this.height - this.x_axis_padding, this.position.y])
        }
        this.sizeChangedListeners.forEach(c => c());
    }

    drawAxes() {
        if (this.draw_x_axis) {
            const y =  this.y.range()[this.invert_y ? 1 : 0];
            drawAxisCompact(this.context, this.x, y, this.x_axis_padding, true);
            if (this.x_axis_label) {
                this.context.textAlign = "right";
                const textMetrics = this.context.measureText(this.x_axis_label);
                const x = this.x.range()[1];
                this.context.clearRect(x - textMetrics.width - 3, y - textMetrics.fontBoundingBoxAscent, textMetrics.width + 3, textMetrics.fontBoundingBoxAscent + textMetrics.fontBoundingBoxDescent);
                this.context.fillText(this.x_axis_label, x, y);
            }
        }
        if (this.draw_y_axis) drawAxis(this.context, this.y, this.x.range()[0], this.y_axis_padding, false);
    }

    clear() {
        this.context.clearRect(this.position.x, this.position.y - 1, this.width, this.height);
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

    // function drawMenuBar(menu_bar_container) {
    //     // View
    //     const view = menu_bar_container.append("button").text("View")
    //     const view_menu = menu_bar_container.append("div").attr("class", "dropdown-content");
    //     view_menu.append("a").text("item1");
    //     view_menu.append("a").text("item2");
    //     view.on("click", () => view_menu.classed("show", true));
            
    // }

    function drawHelpButton(container) {
        const pill = container.append("a").attr("href", "https://github.com/fgnt/meeteval").classed("pill", true)
        pill.append("i").classed("fas fa-question-circle", true);
        pill.append("div").text("Help");
    }

    function drawExampleInfo(container, info) {
        const root = container; //container.append("div").classed("info-container", true);

        label = (label, value) => {
            var l = root.append("div").classed("pill", true);
            l.append("div").classed("info-label", true).text(label);
            l.append("div").classed("info-value", true).text(value);
            return l;
        }

        label("ID:", info.filename);
        label("Length:", info.length + "s");
        label("WER:", (info.wer.error_rate * 100).toFixed(2) + "%");
        label("Alignment:", info.alignment_type)
        if (info.wer.reference_self_overlap?.overlap_rate) label("Reference self-overlap:", (info.wer.reference_self_overlap.overlap_rate * 100).toFixed(2) + "%").classed("warn", true);
        if (info.wer.hypothesis_self_overlap?.overlap_rate) label("Hypothesis self-overlap:", (info.wer.hypothesis_self_overlap.overlap_rate * 100).toFixed(2) + "%").classed("warn", true);
    }

    class ErrorBarPlot {
        constructor(canvas_plot, num_bins, words, style='absolute', scaleExcludeCorrect=false) {
            this.plot = canvas_plot;
            this.bin = d3.bin().thresholds(200).value(d => (d.begin_time + d.end_time) / 2)
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
                d.substitutions = d.map(w => w.match_type === 'substitution').reduce((a, b) => a + b, 0);
                d.insertions = d.map(w => w.match_type === 'insertion').reduce((a, b) => a + b, 0);
                d.deletions = d.map(w => w.match_type === 'deletion').reduce((a, b) => a + b, 0);
                d.total = d.length;

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
            this.plot.context.strokeStyle = "black";
            this.words.filter(d => d.begin_time < end && d.end_time > begin).forEach(u => {
                this.plot.context.beginPath();
                let y_;
                if (u.source === "hypothesis") {
                    y_ = this.plot.y(u.speaker_id) + this.plot.y.bandwidth() / 2;
                } else {
                    y_ = this.plot.y(u.speaker_id);
                }
                this.plot.context.rect(
                    this.plot.x(u.begin_time),
                    y_,
                    this.plot.x(u.end_time) - this.plot.x(u.begin_time),
                    this.plot.y.bandwidth() / 2,
                );
                if (u.match_type !== undefined) {
                    this.plot.context.fillStyle = settings.colors[u.match_type];
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
                    new CanvasPlot(e, width, 40,
                    x_scale,
                    d3.scaleLinear().domain([1, 0]),
                        false, false, true,
                ), 200, words, settings.barplot.style, settings.barplot.scaleExcludeCorrect);
            }
            this.word_plot = new WordPlot(
                new CanvasPlot(e, width, 100, x_scale, y_scale,
                    false, true, true, "time"),
                words
            );

            if (settings.barplot.style !== "hidden") {
                this.error_bars.plot.element.append("div")
                    .style("top", 0).style("left", 0).style("position", "absolute").style("margin-left", this.word_plot.plot.y_axis_padding + "px")
                    .style("padding", "0 3px 0 3px")
                    .style("font-style", "italic").style("user-select", "none")
                    // .style("border", "1px solid gray")
                    .style("border-radius", "0 5px 5px 0")
                    .style("font-size", "10px")
                    .text("Error distribution");
            }

            
            this.word_plot.plot.element.append("div")
                .style("top", 0).style("left", 0).style("position", "absolute").style("margin-left", this.word_plot.plot.y_axis_padding + "px")
                .style("padding", "0 3px 0 3px")
                .style("font-style", "italic").style("user-select", "none")
                // .style("border", "1px solid gray")
                .style("border-radius", "0 5px 5px 0")
                .style("font-size", "10px")
                .text("Utterances");

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

        _onselect(event) {
            if (event.selection === null) {
                this.selection = this.max_range;
            } else {
                this.selection = event.selection;
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
            this.h = new Howl({src: src})
            this.position = 0;
            this.animationFrameID = null;
            this.h.once('end', () => this.remove.bind(this))
            this.audio_data = null
            this.global_context = global_context
            this.global_x_position = global_x_position

            const self = this;
            fetch(src)
                .then(
                    r => r.arrayBuffer().then(
                        b => new AudioContext().decodeAudioData(b).then(
                            a => {
                                self.audio_data = a.getChannelData(0);
                                self.drawAudio();
                            }
                        )
                    )
                )

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
        constructor(plot, words, utterances, alignment, ref_hyp_gap=10) {
            this.plot = plot;
            this.plot.x_axis_padding = 50;
            this.words = words;
            this.filtered_words = words;
            this.utterances = utterances;
            this.filtered_utterances = utterances;
            this.alignment = alignment;
            this.speaker_ids = utterances.map(d => d.speaker_id);
            this.max_length = plot.y.domain()[1];
            this.ref_hyp_gap = ref_hyp_gap;

            this.playhead = null

            const self = this;
            // Create elements for click handlers
            this.plot.element.on("click", (event) => {
                const screenX = event.layerX;
                const screenY = event.layerY;
                const y = self.plot.y.invert(screenY);

                // invert x band scale
                const eachBand = self.plot.x.step();
                const index = Math.floor((screenX - self.plot.y_axis_padding) / eachBand);
                const speaker_id = self.plot.x.domain()[index];

                const utterance_candidates = this.filtered_utterances.filter(
                    u => u.begin_time < y && u.end_time > y && u.speaker_id === speaker_id && u.source === "hypothesis"
                )
                if (utterance_candidates.length > 0) {
                    const utterance = utterance_candidates[0];
                    // TODO: handle utterance click
                    // console.log("click", screenX, index, speaker_id, y, self.plot.x.domain());
                }
            })

            // const container = this.plot.element.append("div")
            //     .style("position", "absolute")
            //     .style("top", 0)
            //     .style("left", 0)
            //     .style("width", "100%")
            //     .style("height", "100%");
            // container.selectAll("div").data(this.plot.x.domain())
            //     .enter()
            //     .append("div")
            //     // .style("background-color", "red")
            //     .style("position", "absolute")
            //     .style("top", 0)
            //     .style("left", d => this.plot.x(d) + "px")
            //     .style("height", "100%")
            //     .style("width", this.plot.x.bandwidth() + "px")
            //     .on("click", this.click.bind(this))
            //     .append("canvas")
            //     .attr("id", d => "canvas-" + d)
            //     .attr("height", this.plot.height)
            //     .attr("width", this.plot.x.bandwidth())
            //     .style("width", "100%")
            //     .style("height", "100%")

            // TODO: allow scrolling and dragging and ctrl+scroll for zoom
            this.plot.element.on("wheel", (event) => {
                let [begin, end] = this.plot.y.domain();
                let delta = (this.plot.y.invert(event.deltaY) - this.plot.y.invert(0)) * 0.3
                if (event.ctrlKey) {
                    begin = Math.max(0, begin - delta);
                    end = Math.min(end + delta, this.max_length);
                } else {
                    if (end + delta > this.max_length) delta = this.max_length - end;
                    if (begin + delta < 0) delta = -begin;
                    begin = begin + delta;
                    end = end + delta;
                }
                // TODO: We shouldn't call zoomTo here because it would create an update loop
                this._callOnScrollHandlers(begin, end);
                event.preventDefault();
            }, false)

            // this.plot.element.on("drag", e => console.log("dragging", e))
            this.plot.element.on("mousemove", event => {
                if (event.buttons !== 1) return;
                const delta = this.plot.y.invert(event.movementY) - this.plot.y.invert(0);
                let [begin, end] = this.plot.y.domain();
                this._callOnScrollHandlers(begin - delta, end - delta);
            })

            this.onscrollhandlers = [];

            this.plot.onSizeChanged(this.draw.bind(this));
        }

        click(event, speaker_id) {
            if (this.playhead !== null) this.playhead.stop();
            const screenY = event.layerY;
            const y = this.plot.y.invert(screenY);
            const utterance_candidates = this.filtered_utterances.filter(
                u => u.begin_time < y && u.end_time > y && u.speaker_id === speaker_id && u.source === "hypothesis"
            )
            if (utterance_candidates.length === 0) return;
            const utterance = utterance_candidates[0];
            // const canvas = d3.select('#canvas-' + speaker_id).node()
            // this.playhead = new PlayHead(
            //     canvas, canvas.getContext("2d"),
            //     this.plot.context,
            //     this.plot.x(speaker_id) + this.plot.x.bandwidth(),
            //     this.plot.x, this.plot.y, utterance.begin_time, utterance.end_time,
            //     // TODO: remove fallback
            //     utterance.audio || 'example.wav'
            // )
            // this.playhead.play();
            // this.draw();
        }

        onScroll(callback) {
            this.onscrollhandlers.push(callback);
        }

        _callOnScrollHandlers(x0, x1) {
            this.onscrollhandlers.forEach(c => c(x0, x1));
        }

        drawYAxisLabels() {
            const ticks = this.plot.x.domain().map(d => {
                return {
                    pos: this.plot.x(d) + (this.plot.x.bandwidth ? this.plot.x.bandwidth() / 2 : 0),
                    label: d
                }
            })
            const position = this.plot.y.range()[1];
            const tickSize = 6;
            const offset = (this.plot.x.bandwidth()) / 4;
            const tickPadding = 3;
            this.plot.context.textAlign = "center";
            this.plot.context.textBaseline = "top";
            const Y = position;
            ticks.forEach(d => {
                this.plot.context.beginPath();
                this.plot.context.rect(d.pos - this.plot.x.bandwidth() / 2, Y, this.plot.x.bandwidth(), this.plot.x_axis_padding);
                this.plot.context.fillStyle = "white";
                this.plot.context.strokeStyle = "black";
                this.plot.context.fill();
                this.plot.context.stroke();
                this.plot.context.beginPath();
                this.plot.context.moveTo(d.pos - this.ref_hyp_gap, Y);
                this.plot.context.lineTo(d.pos - this.ref_hyp_gap, Y + tickSize);
                this.plot.context.moveTo(d.pos + this.ref_hyp_gap, Y);
                this.plot.context.lineTo(d.pos + this.ref_hyp_gap, Y + tickSize);
                this.plot.context.moveTo(d.pos + this.plot.x.bandwidth() / 2, Y);
                this.plot.context.lineTo(d.pos + this.plot.x.bandwidth() / 2, Y + tickSize);
                this.plot.context.moveTo(d.pos - this.plot.x.bandwidth() / 2, Y);
                this.plot.context.lineTo(d.pos - this.plot.x.bandwidth() / 2, Y + tickSize);
                this.plot.context.stroke();
                this.plot.context.fillStyle = "black";
                this.plot.context.fillText(d.label, d.pos, position + tickSize + tickPadding)
                this.plot.context.fillText("REF", d.pos - offset, Y + tickPadding);
                this.plot.context.fillText("HYP", d.pos + offset, Y + tickPadding);
            });
        }

        drawDetails() {
            const [begin, end] = this.plot.y.domain();

            const filtered_words = this.filtered_words;
            if (filtered_words.length > 3000) return;
            const filtered_utterances = this.filtered_utterances;
            const context = this.plot.context;

            const draw_text = filtered_words.length < 400;
            const draw_boxes = filtered_words.length < 1000;
            const draw_utterance_markers = filtered_words.length < 2000;
            const band_width = this.plot.x.bandwidth() / 2 - this.ref_hyp_gap;

            // Draw words
            context.font = `${settings.font_size}px Arial`;
            context.textAlign = "center";
            context.textBaseline = "middle";
            context.lineWidth = 1;
            filtered_words.forEach(d => {
                let x_;
                if (d.source === "hypothesis") {
                    x_ = this.plot.x(d.speaker_id) + this.plot.x.bandwidth() / 2 + this.ref_hyp_gap;
                } else {
                    x_ = this.plot.x(d.speaker_id);
                }

                context.beginPath();
                context.rect(
                    x_,
                    this.plot.y(d.begin_time),
                    band_width,
                    this.plot.y(d.end_time) - this.plot.y(d.begin_time));
                context.strokeStyle = 'gray';
                context.fillStyle = settings.colors[d.match_type];
                context.fill();
                if (draw_boxes) context.stroke();

                // Text
                if (draw_text) {
                    x_ += band_width / 2;
                    let y_ = this.plot.y((d.begin_time + d.end_time) / 2);

                    context.fillStyle = '#000';
                    context.fillText(d.transcript, x_, y_);
                }
            })

            // Draw stitches
            const filtered_alignment = this.alignment.filter(d => {
                const begin_time = d.ref_center_time === undefined || d.ref_center_time > d.hyp_center_time ? d.hyp_center_time : d.ref_center_time;
                const end_time = d.ref_center_time === undefined || d.ref_center_time < d.hyp_center_time ? d.hyp_center_time : d.ref_center_time;
                return begin_time < end && end_time > begin;
            });
            const lineStartOffset = this.ref_hyp_gap / 8;
            context.lineWidth = 3;
            filtered_alignment.forEach(d => {
                const x_ref = this.plot.x(d.ref_speaker_id) + this.plot.x.bandwidth() / 2;
                const x_hyp = this.plot.x(d.hyp_speaker_id) + this.plot.x.bandwidth() / 2;
                context.beginPath();
                context.strokeStyle = settings.colors[d.match_type];
                if (d.hyp_center_time === undefined) {
                    const y = this.plot.y(d.ref_center_time);
                    context.moveTo(x_ref - this.ref_hyp_gap, y);
                    context.lineTo(x_hyp, y);
                } else if (d.ref_center_time === undefined) {
                    const y = this.plot.y(d.ref_center_time);
                    context.moveTo(x_ref, y);
                    context.lineTo(x_hyp + this.ref_hyp_gap, y);
                } else {
                    const xl = x_ref - this.ref_hyp_gap;
                    const yl = this.plot.y(d.ref_center_time)
                    const xr = x_hyp + this.ref_hyp_gap;
                    const yr = this.plot.y(d.hyp_center_time)
                    context.moveTo(xl - lineStartOffset, yl);
                    context.lineTo(xl + lineStartOffset, yl);
                    context.lineTo(xr - lineStartOffset, yr);
                    context.lineTo(xr + lineStartOffset, yr);
                }
                context.stroke();
            });

            // Draw utterance begin and end markers
            const markerLength = 6;
            const markerOverhang = 3;
            if (draw_utterance_markers) {
                filtered_utterances.forEach(d => {
                    context.strokeStyle = "black";
                    context.lineWidth = 2;
                    context.beginPath();

                    // x is the left side of the marker
                    var x = this.plot.x(d.speaker_id);
                    const bandwidth = this.plot.x.bandwidth() / 2 - this.ref_hyp_gap;
                    if (d.source == "hypothesis") {
                        x += bandwidth + 2*this.ref_hyp_gap;
                    }

                    var y = this.plot.y(d.begin_time) - 1;

                    context.moveTo(x - markerOverhang, y + markerLength);
                    context.lineTo(x - markerOverhang, y);
                    context.lineTo(x + bandwidth + markerOverhang, y);
                    context.lineTo(x + bandwidth + markerOverhang, y + markerLength);

                    y = this.plot.y(d.end_time) + 1;
                    context.moveTo(x - markerOverhang, y - markerLength);
                    context.lineTo(x - markerOverhang, y);
                    context.lineTo(x + bandwidth + markerOverhang, y);
                    context.lineTo(x + bandwidth + markerOverhang, y - markerLength);
                    context.stroke();
                    
                    // Draw marker that text is empty
                    if (d.transcript === "") {
                        context.beginPath();
                        context.strokeStyle = "lightgray";
                        context.linewidth = 1;
                        x = x + bandwidth / 2;
                        // context.moveTo(x, this.plot.y(d.begin_time));
                        // context.lineTo(x, this.plot.y(d.end_time) );
                        // context.stroke();
                        context.font = `italic ${settings.font_size}px Arial`;
                        context.fillStyle = "gray";
                        context.fillText('(empty segment)', x, this.plot.y((d.begin_time + d.end_time) / 2));
                    }
                    
                });
            }
        }

        draw() {
            this.plot.clear();
            this.drawDetails();
            if (this.playhead !== null) this.playhead.drawAudio();
            this.plot.drawAxes();
            this.drawYAxisLabels();
        }

        zoomTo(x0, x1) {
            this.plot.y.domain([x0, x1]);
            this.filtered_words = this.words.filter(w => w.begin_time < x1 && w.end_time > x0);
            this.filtered_utterances = this.utterances.filter(w => w.begin_time < x1 && w.end_time > x0);

            call_throttled(this.draw.bind(this));
        }
    }


    // Data preprocessing
    const utterances = data.utterances;
    const words = data.words;
    const alignment = data.alignment;

    const time_domain = [0, Math.max.apply(null, (utterances.map(d => d.end_time))) + 1];
    const speaker_ids = utterances
        // .filter(u => u.source === 'reference')
        .map(d => d.speaker_id)

    // Setup plot elements
    var margin = {top: 30, right: 30, bottom: 70, left: 60},
        width = 1500 - margin.left - margin.right,
        height = 300 - margin.top - margin.bottom;
    const top_row_container = d3.select(element_id).append("div").classed("top-row", true)
    drawHelpButton(top_row_container);
    drawExampleInfo(top_row_container, data.info)
    // drawMenuBar(top_row_container);
    if (settings.show_legend) drawLegend(top_row_container);
    drawMenu(top_row_container);
    const plot_container = d3.select(element_id).append("div").style("margin", "10px")
    const plot_div = plot_container.append("div").style("position", "relative")

    var minimaps = []
    var details_plot = null;
    function rebuild() {
        plot_div.selectAll("*").remove();
        minimaps = [];
        details_plot = null;

        for (let i = 0; i < settings.minimaps.number; i++) {
            const minimap = new Minimap(
                plot_div, width, settings.minimaps.height,
                d3.scaleLinear().domain(time_domain),
                d3.scaleBand().domain(speaker_ids).padding(0.1),
                words,
            )
            if (minimaps[i-1] !== undefined) {
                minimaps[i-1].onSelect(minimap.zoomTo.bind(minimap));
            }
            minimaps.push(minimap);
        }

        if (settings.show_details) {
            details_plot = new DetailsPlot(
                new CanvasPlot(plot_div, width, 700,
                    d3.scaleBand().domain(speaker_ids).padding(0.1),
                    d3.scaleLinear().domain([time_domain[0], time_domain[1]]),
                    true
                ), words, utterances, alignment
            )


            if (minimaps.length > 0) {
                const last_minimap = minimaps[minimaps.length - 1];
                last_minimap.onSelect(details_plot.zoomTo.bind(details_plot));
                details_plot.onScroll((x0, x1) => {
                    last_minimap.brush_group.call(last_minimap.brush.move, [
                        last_minimap.word_plot.plot.x(x0), last_minimap.word_plot.plot.x(x1)
                    ])
                });
            } else {
                // This is necessary to prevent update loops. We can't call details_plot.zoomTo in details_plot...
                details_plot.onScroll(details_plot.zoomTo.bind(details_plot));
            }
        }
    }

    function redraw() {
        if (settings.show_details) details_plot.draw();
        for (const minimap of minimaps) minimap.draw();
    }

    rebuild();
    redraw();
}