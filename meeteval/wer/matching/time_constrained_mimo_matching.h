#include <vector>
#include <numeric>
#include <stdio.h>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <memory>
#include <cassert>
#include <stdexcept>
#include <limits>

/*
* A hash function for vectors so that they can be used as keys in unordered_map.
*
* This might change in the future. We might switch to a plain integer representation for indexes, similar to the
* (not-time-constrained) mimo algorithm.
*
* https://stackoverflow.com/a/53283994
*/
struct VectorHasher {
    int operator()(const std::vector<unsigned int> &V) const {
        int hash = V.size();
        for(auto &i : V) {
            hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

/*
 * Get the "next" index. The next index is one that is one where exactly one index is larger than the current index.
 * If the index is already the last index, it is reset to the first index.
 * Returns true if the index was advanced, false if it was reset
 */
bool advance_index(
    std::vector<unsigned int> & index,
    const std::vector<unsigned int> & dimensions,
    const unsigned int ignore_index=-1
) {
    assert(index.size() == dimensions.size());
    for (std::vector<unsigned int>::size_type d = 0; d < dimensions.size(); d++) {
        assert(index.at(d) < dimensions.at(d));
        if (d == ignore_index) continue;
        if (index.at(d) == dimensions.at(d) - 1) {
            index.at(d) = 0;
        } else {
            index.at(d)++;
            return true;
        }
    }
    return false;
}

/*
 * Represents a memory layout of a tensor when stored flattened in a vector.
 *
 * get_index() returns the (integer) index of a given (vector) index in the flattened vector.
 * advance_index() advances the index to the next index according to the dimensions of this layout. Returns true if
 *  the index was advanced, false if it was reset
 * within() checks if the given index is within the dimensions of this layout
 */
struct Layout {
    std::vector<unsigned int> strides;
    std::vector<unsigned int> dimensions;
    size_t total_size;

    unsigned int get_index(const std::vector<unsigned int> & index) const {
        assert(index.size() == dimensions.size());
        size_t i = 0;
        for (size_t d = 0; d < dimensions.size(); d++) {
            assert(index.at(d) < dimensions.at(d));
            i += index.at(d) * strides.at(d);
        }
        return i;
    }

    bool advance_index(std::vector<unsigned int> & index, const unsigned int ignore_index=-1) const {
        return ::advance_index(index, dimensions, ignore_index);
    }

    bool within(const std::vector<unsigned int> & index) const {
        assert(index.size() == dimensions.size());
        for (size_t d = 0; d < dimensions.size(); d++) {
            if (index.at(d) >= dimensions.at(d)) return false;
        }
        return true;
    }
};

/*
 * Represents a path through the reference state space
 */
struct Path {
    std::shared_ptr<Path> previous;
    unsigned int speaker;
    unsigned int utterance;
    unsigned int stream;
};

/*
 * One entry in the hypothesis state space
 *
 * cost: The cost of the path to this state
 * path: The path (in the reference space) to this state
 */
struct HypStateEntry {
    std::shared_ptr<Path> path;
    unsigned int cost;

    /*
     * Returns a new HypStateEntry with the cost incremented by the given value
     */
    HypStateEntry incremented_by(const unsigned int increment) const {
        return HypStateEntry{path, cost + increment};
    }
};

/*
 * Represents an entry state in the reference space
 *
 * cost: The cost (in the hypothesis space) of the best path to this state for that hypothesis
 * layout: The layout of `cost` in the hypothesis space
 * offset: The offset of `cost` in the hypothesis space
 */
struct StateEntry {
    std::vector<HypStateEntry> cost;
    Layout layout;
    std::vector<unsigned int> offset;
};

/*
 * Creates a `Layout` from a begin pointer and an end pointer.
 */
Layout inline make_layout(
    const std::vector<unsigned int> begin_pointer,
    const std::vector<unsigned int> end_pointer
) {
    Layout layout;
    for (size_t d = 0; d < begin_pointer.size(); d++) {
        layout.dimensions.push_back(end_pointer[d] - begin_pointer[d] + 1);
    }
    size_t j = 1;
    for (auto i : layout.dimensions) {
        layout.strides.push_back(j);

        // Check for overflow before multiplying
        if (j * i / i  != j) throw std::overflow_error("overflow_error");
        j *= i;
    }
    layout.total_size = j;
    return layout;
}

/*
 * Represents one reference utterance
 *
 * begin_time: The begin time of the utterance (earliest begin time of all words)
 * end_time: The end time of the utterance (latest end time of all words)
 */
struct Utterance {
    double begin_time;
    double end_time;
    double overlapping_hyp_begin_time;
    double overlapping_hyp_end_time;
};

/*
 * Bundles all timing information needed for one word
 */
struct Timing {
    double begin_time;
    double end_time;
    double latest_begin_time;
    double earliest_end_time;
};

/*
 * Checks if two words overlap in time
 */
bool overlaps(const Timing & a, const Timing & b) {
    return a.begin_time < b.end_time && b.begin_time < a.end_time;
}

/*
 * Creates a std::vector<Timing> from a std::vector<std::pair<double, double>>. Copies the timings
 * into the struct and fills the latest_begin_time and earliest_end_time fields.
 */
std::vector<Timing> make_extended_timing(const std::vector<std::pair<double, double>> & timings) {
    std::vector<Timing> extended_timings(timings.size());
    if (timings.size() == 0) return extended_timings;

    double end_time = 0;
    for (size_t i = 0; i < timings.size(); i++) {
        end_time = std::max(end_time, timings.at(i).second);
        extended_timings.at(i) = {
            .begin_time=timings.at(i).first,
            .end_time=timings.at(i).second,
            // latest_begin_time is filled below
            .earliest_end_time=end_time,
        };
    }

    double begin_time = end_time;
    for (size_t i = timings.size() - 1;; --i) {
        begin_time = std::min(begin_time, timings.at(i).first);
        extended_timings.at(i).latest_begin_time = begin_time;
        if (i == 0) break;
    }
    return extended_timings;
}

/*
 * Struct to hold the state of the update algorithm
 */
struct UpdateState {
    unsigned int cost;
    unsigned int index;
};

/*
 * All possible match types. Used to precompute pairwise matches between reference and hypothesis words.
 */
enum MatchType {
    NO_EXTENDED_OVERLAP_REF_BEFORE_HYP = 0,
    NO_EXTENDED_OVERLAP_HYP_BEFORE_REF = 1,
    LOCAL_OVERLAP_CORRECT = 2,
    LOCAL_OVERLAP_INCORRECT = 3,
    LOCAL_NO_OVERLAP = 4
};

/*
 * Get the `MatchType` of a pair of words
 */
MatchType get_match_type(const unsigned int ref_symbol, const unsigned int hyp_symbol, const Timing & ref_timing, const Timing & hyp_timing) {
    if (hyp_timing.earliest_end_time <= ref_timing.latest_begin_time) return MatchType::NO_EXTENDED_OVERLAP_HYP_BEFORE_REF;
    else if (ref_timing.earliest_end_time <= hyp_timing.latest_begin_time) return MatchType::NO_EXTENDED_OVERLAP_REF_BEFORE_HYP;
    else if (!overlaps(ref_timing, hyp_timing)) return MatchType::LOCAL_NO_OVERLAP;
    else if (ref_symbol == hyp_symbol) return MatchType::LOCAL_OVERLAP_CORRECT;
    else return MatchType::LOCAL_OVERLAP_INCORRECT;
}

/*
 * Precomputes the match types for all pairs of words in the reference and hypothesis utterances.
 *
 * This speeds up the hot loop significantly.
 */
void precompute_matches(
    const std::vector<std::vector<std::vector<unsigned int>>> reference,
    const std::vector<std::vector<unsigned int>> hypothesis,
    const std::vector<std::vector<std::vector<Timing>>> & reference_timings,
    const std::vector<std::vector<Timing>> & hypothesis_timings,
    // Return types
    std::vector<std::vector<MatchType>> & match_types,
    std::vector<std::vector<std::vector<unsigned int>>> & reference_match_index,
    std::vector<std::vector<unsigned int>> & hypothesis_match_index
) {
    unsigned int reference_index = 0;
    unsigned int hypothesis_index = 0;

    // Iterate through references
    for (size_t speaker = 0; speaker < reference.size(); speaker++) {
        reference_match_index.push_back(std::vector<std::vector<unsigned int>>());
        for (size_t u = 0; u < reference[speaker].size(); u++) {
            reference_match_index.back().push_back(std::vector<unsigned int>());
            for (size_t rw = 0; rw < reference[speaker][u].size(); rw++) {
                reference_match_index.back().back().push_back(reference_index);
                hypothesis_index = 0;
                match_types.push_back(std::vector<MatchType>());

                // Iterate through hypotheses
                for (size_t stream = 0; stream < hypothesis.size(); stream++) {
                    if (reference_index == 0) { hypothesis_match_index.push_back(std::vector<unsigned int>()); }
                    for (size_t hw = 0; hw < hypothesis[stream].size(); hw++) {
                        if (reference_index == 0) { hypothesis_match_index.back().push_back(hypothesis_index); }
                        match_types.back().push_back(get_match_type(
                            reference[speaker][u][rw],
                            hypothesis[stream][hw],
                            reference_timings[speaker][u][rw],
                            hypothesis_timings[stream][hw]
                        ));
                        hypothesis_index++;
                    }
                }
                reference_index++;
            }
        }
    }
}

/*
 * Computes a time-constrained update of a levenshtein row, in-place.
 */
void update_levenshtein_row(
    std::vector<UpdateState> &row,
    const std::vector<unsigned int> & reference_indices,
    const std::vector<unsigned int> & hypothesis_indices,
    const std::vector<std::vector<MatchType>> & match_types,
    const unsigned int hypothesis_begin,
    const unsigned int hypothesis_end
) {
    assert(row.size() >= hypothesis_end - hypothesis_begin + 1);
//    assert(reference.size() == reference_timings.size());
//    assert(hypothesis.size() == hypothesis_timings.size());
    assert(hypothesis_end <= hypothesis_indices.size());
    assert(hypothesis_begin <= hypothesis_end);

    unsigned int steps = hypothesis_end - hypothesis_begin;

    // These two variables track the currently "active" region in the hypothesis
    // Any word before hyp_start and after hyp_end will be ignored (as not overlapping)
    // hyp_start and hyp_end are updated in every iteration
    unsigned int hyp_start = 0;
    unsigned int hyp_end = 0;

    for (size_t reference_index = 0; reference_index < reference_indices.size(); reference_index++) {
        auto & ref_match_types = match_types.at(reference_indices[reference_index]);

        UpdateState diagonal = row[hyp_start];
        row[hyp_start].cost++;
        UpdateState left = row[hyp_start];
        unsigned int row_index = hyp_start + 1;
        bool allow_shift = true;
        for (; row_index < steps + 1; row_index++ ) {
            MatchType match_type = ref_match_types.at(hypothesis_indices[row_index + hypothesis_begin - 1]);

            if (allow_shift) {
                if (match_type == MatchType::NO_EXTENDED_OVERLAP_HYP_BEFORE_REF) {
                    // We can shift the starting point forward because nothing will overlap with anything before hyp_start
                    hyp_start++;
                } else allow_shift = false;
            }

            // Get cost from above. If we are at the end of the row, we have to add the distance to the top (which is
            // reference_index) and consider the path from the left (which is now diagonal)
            UpdateState up = row[row_index];
            if (row_index > hyp_end) {
                up.cost += reference_index;
                if (diagonal.cost < up.cost) {
                    up = diagonal;
                    up.cost++;
                }
            }

            switch (match_type) {
                case MatchType::NO_EXTENDED_OVERLAP_HYP_BEFORE_REF:
                case MatchType::NO_EXTENDED_OVERLAP_REF_BEFORE_HYP:
                case MatchType::LOCAL_NO_OVERLAP:
                    // No overlap. No substitution/correct allowed
                    if (up.cost < left.cost) left = up;
                    left.cost++;
                    break;
                case MatchType::LOCAL_OVERLAP_CORRECT:
                    // Overlap & correct: Diagonal is always the best path
                    left = diagonal;
                    break;
                case MatchType::LOCAL_OVERLAP_INCORRECT:
                    // Overlap & incorrect: Find the best path
                    if (up.cost < left.cost) left = up;
                    if (diagonal.cost < left.cost) left = diagonal;
                    left.cost++; // Cost for ins/del/sub = 1
                    break;
                default:
                    // Can never happen
                    assert(false);
            }

            row[row_index] = left;
            diagonal = up;

            if (allow_shift) {
                // The final value for the cell at hyp_start can only be reached by insertions (direct path down)
                // from the current row. So, we can compute the final value by adding the number of updates
                // that come after this
                row[row_index - 1].cost += reference_indices.size() - reference_index - 1;
            }
            if (match_type == MatchType::NO_EXTENDED_OVERLAP_REF_BEFORE_HYP) {
                break;
            }
        }
        hyp_end = row_index;
    }

    for (; hyp_end < steps; hyp_end++) {
        // Get update from top: Initial value + reference.size() insertions
        UpdateState up = row[hyp_end + 1];
        up.cost += reference_indices.size();

        // Get update from left: left value + 1 deletion
        UpdateState left = row[hyp_end];
        left.cost++;

        if (up.cost < left.cost) left = up;
        row[hyp_end + 1] = left;
    }

}


std::vector<Utterance> make_utterances(
    std::vector<std::vector<std::pair<double, double>>> & timings,
    std::vector<std::vector<std::pair<double, double>>> & hypothesis_timings
) {
    std::vector<Utterance> utterances(timings.size());
    if (timings.size() == 0) return utterances;
    
    // Flatten and sort timings
    std::vector<std::pair<double, double>> flat_hyp_timings;
    for (auto &v : hypothesis_timings) {
        flat_hyp_timings.insert(flat_hyp_timings.end(), v.begin(), v.end());
    }
    auto flat_hyp_timings_sorted_begin = flat_hyp_timings;
    std::sort(flat_hyp_timings_sorted_begin.begin(), flat_hyp_timings_sorted_begin.end(), [](std::pair<double, double> a, std::pair<double, double> b) {
        return a.first < b.first;
    });
    auto flat_hyp_timings_sorted_end = flat_hyp_timings;
    std::sort(flat_hyp_timings_sorted_end.begin(), flat_hyp_timings_sorted_end.end(), [](std::pair<double, double> a, std::pair<double, double> b) {
        return a.second < b.second;
    });
    
    double end_time = 0;
    for (size_t i = 0; i < timings.size(); i++) {
        // Handle case where words are not sorted by time
         double begin = std::min_element(
            timings[i].begin(),
            timings[i].end(),
            [](const std::pair<double, double> & a, const std::pair<double, double> & b) {
                return a.first < b.first;
            }
        )->first;
        end_time = std::max(std::max_element(
            timings[i].begin(),
            timings[i].end(),
            [](const std::pair<double, double> & a, const std::pair<double, double> & b) {
                return a.second < b.second;
            })->second, end_time
        );

        utterances[i] = {
            .begin_time=begin,
            .end_time=end_time,
            .overlapping_hyp_begin_time=std::numeric_limits<double>::max(),
            .overlapping_hyp_end_time=std::numeric_limits<double>::min()
        };
    }

    double begin_time = end_time;
    for (size_t i = utterances.size() - 1;; --i) {
        begin_time = std::min(begin_time, utterances[i].begin_time);
        utterances[i].begin_time = begin_time;

        // Find the earliest begin time of the overlapping hypothesis words in flat_hyp_timings
        auto it = std::lower_bound(flat_hyp_timings_sorted_begin.begin(), flat_hyp_timings_sorted_begin.end(), begin_time, [](std::pair<double, double> a, double b) {
            return a.second < b;
        });
        if (it != flat_hyp_timings_sorted_begin.end()) {
            utterances[i].overlapping_hyp_begin_time = it->first;
        }

        // Find the latest end time fo the overlapping hypothesis words in flat_hyp_timings
        it = std::lower_bound(flat_hyp_timings_sorted_end.begin(), flat_hyp_timings_sorted_end.end(), end_time, [](std::pair<double, double> a, double b) {
            return b < a.first;
        });
        if (it != flat_hyp_timings_sorted_end.begin()) {
            it--;
            utterances[i].overlapping_hyp_end_time = it->second;
        }

        if (i == 0) break;
    }
    assert(utterances.size() == timings.size());
    return utterances;
}


/*
 * This implementation is _not_ error tolerant. Specifically, the input must conform to the following constraints:
 *  - Utterances must be sorted for this algorithm to work
 *  - All utterances must contain at least one word
 *  - Must have at least one speaker and one stream
 */
std::pair<unsigned int, std::vector<std::pair<unsigned int, unsigned int>>> time_constrained_mimo_levenshtein_distance_(
    std::vector<std::vector<std::vector<unsigned int>>> reference,
    std::vector<std::vector<unsigned int>> hypothesis,
    std::vector<std::vector<std::vector<std::pair<double, double>>>> reference_timings,
    std::vector<std::vector<std::pair<double, double>>> hypothesis_timings
) {
    // Check inputs
    assert(reference.size() == reference_timings.size());
    assert(hypothesis.size() == hypothesis_timings.size());
    for (unsigned int i = 0; i < reference.size(); i++) assert(reference[i].size() == reference_timings[i].size());
    for (unsigned int i = 0; i < hypothesis.size(); i++) assert(hypothesis[i].size() == hypothesis_timings[i].size());
    size_t num_reference_speakers = reference.size();
    size_t num_hypothesis_streams = hypothesis.size();
    unsigned int num_utterances = 0;
    for (size_t d = 0; d < num_reference_speakers; d++) {
        num_utterances += reference[d].size();
    }

    // Compute for every word in the hypothesis the earliest seen end time and the latest seen begin time
    // This is required when the words are not sorted by begin time.
    std::vector<std::vector<Timing>> extended_hypothesis_timings(num_hypothesis_streams);
    for (size_t s = 0; s < hypothesis.size(); s++) {
        extended_hypothesis_timings.at(s) = make_extended_timing(hypothesis_timings.at(s));
    }

    // Convert reference utterances into `Utterance` structs
    std::vector<std::vector<Utterance>> reference_utterances(num_reference_speakers);
    for (size_t d = 0; d < num_reference_speakers; d++) {
        reference_utterances.at(d) = make_utterances(reference_timings.at(d), hypothesis_timings);
    }

    // Compute extended timings for the reference utterances
    std::vector<std::vector<std::vector<Timing>>> extended_reference_timings(num_reference_speakers);
    for (size_t d = 0; d < num_reference_speakers; d++) {
        extended_reference_timings.at(d).resize(reference_timings.at(d).size());
        for (size_t u = 0; u < reference_timings.at(d).size(); u++) {
            extended_reference_timings.at(d).at(u) = make_extended_timing(reference_timings.at(d).at(u));
        }
    }

    // Precompute matches
    std::vector<std::vector<std::vector<unsigned int>>> reference_match_index;
    std::vector<std::vector<unsigned int>> hypothesis_match_index;
    std::vector<std::vector<MatchType>> matches;
    precompute_matches(
        reference,
        hypothesis,
        extended_reference_timings,
        extended_hypothesis_timings,
        matches,
        reference_match_index,
        hypothesis_match_index
    );

    // State: track the currently active region in the reference (i.e., where words overlap)
    // Indices of the first and last utterance in the currently active region for each (reference) stream
    double ref_block_begin_time = 0;
    double ref_block_end_time = 0;
    std::vector<unsigned int> begin_pointer(num_reference_speakers);
    std::vector<unsigned int> end_pointer(num_reference_speakers);

    // State
    std::unordered_map<std::vector<unsigned int>, StateEntry, VectorHasher> state = {{
        std::vector<unsigned int>(num_reference_speakers),
        {
            .cost=std::vector<HypStateEntry>(1),
            .layout=make_layout(std::vector<unsigned int>(num_hypothesis_streams), std::vector<unsigned int>(num_hypothesis_streams)),
            .offset=std::vector<unsigned int>(hypothesis.size())
        }
    }};

    // State: track the currently active region in the hypothesis (i.e., where words overlap)
    // Indices of the first and last word in the currently active region for each (hypothesis) stream
    std::vector<unsigned int> new_state_offset(hypothesis.size());
    std::vector<unsigned int> new_state_end(hypothesis.size());
    
    // Pre-allocate temporary memory
    unsigned int max_num_states = 0;
    for (size_t s = 0; s < hypothesis.size(); s++) max_num_states = std::max(max_num_states, (unsigned int) hypothesis.at(s).size());
    std::vector<UpdateState> tmp_row(max_num_states + 1);
    std::vector<unsigned int> indices;

    // Iterate through reference utterances in temporal order
    unsigned int iterations = 0;
    while (true) {
        iterations++;

        // Forward one reference utterance. Pick the one with the earliest begin time.
        size_t current_speaker_index = std::numeric_limits<size_t>::max();
        for (size_t d = 0; d < num_reference_speakers; d++) {
            if (end_pointer.at(d) >= reference_utterances.at(d).size()) continue;   // already at end of this stream
            if (current_speaker_index == std::numeric_limits<size_t>::max()) {
                current_speaker_index = d;
            } else {
                auto a = reference_utterances.at(d).at(end_pointer.at(d)).begin_time;
                auto b = reference_utterances.at(current_speaker_index).at(end_pointer.at(current_speaker_index)).begin_time;
                if (a < b) current_speaker_index = d;
            }
        }
        auto current_speaker = reference_utterances.at(current_speaker_index);

        // Find all reference utterances that overlap with the current block
        // and have an earlier start point.
        // This is represented for each stream by begin_pointer[stream_index]
        // and end_pointer[stream_index]
        for (size_t d = 0; d < num_reference_speakers; d++) {
            while (
                // We are before the current utterance (represented by end_pointer on the current stream)
                begin_pointer.at(d) < end_pointer.at(d) 
                // We are not at the end of the stream
                && begin_pointer.at(d) < reference_utterances.at(d).size()
                // And both utterances do not overlap with a common hypothesis word
                && reference_utterances.at(d).at(begin_pointer.at(d)).overlapping_hyp_end_time
                < current_speaker.at(end_pointer.at(current_speaker_index)).begin_time
                && reference_utterances.at(d).at(begin_pointer.at(d)).end_time
                < current_speaker.at(end_pointer.at(current_speaker_index)).overlapping_hyp_begin_time
            ) {
                // Prune state: remove anything that is no longer needed (outside of current block of overlapping
                // reference utterances)
                // This reduces memory usage and speeds up state access
                auto index = begin_pointer;
                while (true) {
                    state.erase(index);

                    // Increment index by one
                    unsigned int d_ = 0;
                    for (; d_ < num_reference_speakers; d_++) {
                        if (index[d_] == end_pointer[d_] || d_ == d) {
                            index[d_] = begin_pointer[d_];
                        } else {
                            index[d_]++;
                            break;
                        }
                    }
                    if (d_ >= num_reference_speakers) break;
                }

                // Forward begin pointer
                begin_pointer.at(d)++;
            }
        }
        // update end poiner to include the current utterance
        end_pointer.at(current_speaker_index)++;

        // Track begin and end time of the current "block" of overlapping reference utterances
        auto p = current_speaker.at(end_pointer.at(current_speaker_index) - 1).end_time;
        ref_block_end_time = std::max(p, ref_block_end_time);
        ref_block_begin_time = -1;
        for (size_t d = 0; d < num_reference_speakers; d++) {
            if (begin_pointer.at(d) >= reference_utterances.at(d).size()) continue;
            auto b = reference_utterances.at(d).at(begin_pointer.at(d)).begin_time;
            if (ref_block_begin_time == -1) ref_block_begin_time = b;
            else if (b < ref_block_begin_time) ref_block_begin_time = b;
        }
        assert(ref_block_begin_time != -1);

        // Sanity checks
        for (unsigned int d = 0; d < num_reference_speakers; d++) {
            assert(begin_pointer.at(d) <= end_pointer.at(d));
            assert(end_pointer.at(d) <= reference_timings.at(d).size());
        }
        assert(state.find(begin_pointer) != state.end());

        // The state at begin_pointer is already computed, 
        // so we can skip computation if it is equal to end_pointer
        if (begin_pointer == end_pointer) continue;

        // Compute size of new state TODO: is this too large?
        // This state has the dimensions of the hypothesis words that overlap
        // on each stream with the current block of reference utterances
        for (size_t s = 0; s < hypothesis.size(); s++) {
            // Find which words overlap with the current reference block, i.e., which portion of the state
            // has to be computed

            while (
                new_state_offset.at(s) < hypothesis.at(s).size()
                && extended_hypothesis_timings.at(s).at(new_state_offset.at(s)).earliest_end_time < ref_block_begin_time
            ) {
                new_state_offset.at(s)++;
            }
            new_state_end.at(s) = 0;
            while (
                new_state_end.at(s) < hypothesis.at(s).size()
                && extended_hypothesis_timings.at(s).at(new_state_end.at(s)).latest_begin_time < ref_block_end_time
            ) {
                new_state_end.at(s)++;
            }
        }

        // Compute new cells
        // We know that everything where index[current_timing.dimension] < end_pointer[current_timing.dimension]
        // is already computed since we are iterating through the utterances in order of their start time
        // and things are monotonic
        std::vector<unsigned int> indices = begin_pointer;
        indices[current_speaker_index] = end_pointer[current_speaker_index];
        for (size_t d = 0; d < indices.size(); d++) assert(indices[d] <= end_pointer[d]);
        while (true) {
            // Assert that the current index is not computed yet
            assert(state.find(indices) == state.end());

            Layout target_layout = make_layout(new_state_offset, new_state_end);
            StateEntry new_state = {
                .cost=std::vector<HypStateEntry>(target_layout.total_size),
                .layout=target_layout,
                .offset=new_state_offset,
            };
            
            // Compute cell at `indices` from previous cells
            auto first_update = true;
            for (unsigned int d = 0; d < indices.size(); d++) {
                if (indices.at(d) == 0) continue;

                // Get previous state. We skip the computation if it doesn't exist because it means that
                // the previous state is not reachable from the current state given the time constraints.
                indices.at(d)--;
                auto prev_state_it = state.find(indices);
                indices.at(d)++;
                if (prev_state_it == state.end()) {
                    continue;
                }
                auto & prev_state = prev_state_it->second;

                // Get words and timings of the currently active reference utterance
                auto &active_reference = reference.at(d).at(indices.at(d) - 1);
                auto &active_reference_timings = extended_reference_timings.at(d).at(indices.at(d) - 1);

                // Compute the update for every hypothesis stream and min over that
                for (unsigned int s = 0; s < hypothesis.size(); s++) {
                    unsigned int old_distance = -1;
                    std::vector<unsigned int> new_state_index(new_state.layout.dimensions.size());
                    std::vector<unsigned int> old_state_index(prev_state.layout.dimensions.size());
                    std::vector<unsigned int> old_closest_index(new_state_index.size());

                    // Iterate over all starting points of of Levenshtein rows along the active_hypothesis_index dimension,
                    // i.e., where new_state_index[s] == 0
                    while(true) {
                        // Translate the new state index into the old state index
                        for (size_t i = 0; i < old_state_index.size(); i++) {
                            assert(new_state.offset.at(i) >= prev_state.offset.at(i));
                            assert(prev_state.offset.at(i) <= new_state_index.at(i) + new_state_offset.at(i));
                            old_state_index.at(i) = new_state_index.at(i) + new_state_offset.at(i) - prev_state.offset.at(i);
                        }

                        // Find the index in prev state that is closest to the new index
                        std::vector<unsigned int> closest_index(new_state_index.size());
                        unsigned int distance = 0;
                        for (size_t i = 0; i < closest_index.size(); i++) {
                            closest_index[i] = std::min(
                                (unsigned int) old_state_index[i],
                                (unsigned int) prev_state.layout.dimensions[i] - 1
                            );
                            assert(old_state_index[i] >= closest_index[i]);
                            distance += old_state_index[i] - closest_index[i];
                        }
                        assert(prev_state.layout.within(closest_index));

                        if (old_distance != -1 && old_closest_index == closest_index && distance > old_distance) {
                            // Shortcut:
                            //
                            // If the distance is greater than 0 we use an earlier state and fill with insertions.
                            // The levenshtein is invariant to a constant offset, so we can simply take the
                            // last row and add the insertions
                            for (unsigned int s_ = 0; s_ < new_state.layout.dimensions.at(s); s_++) {
                                tmp_row[s_].cost += distance - old_distance;
                            }
                        } else {
                            // Fill temporary row
                            for (unsigned int s_ = 0; s_ < new_state.layout.dimensions.at(s); s_++) {
                                if (prev_state.layout.within(closest_index)) {
                                    // Copy from previous state
                                    // We have to add the distance because the states might not be overlapping
                                    tmp_row.at(s_).index = prev_state.layout.get_index(closest_index);
                                    tmp_row.at(s_).cost = prev_state.cost.at(tmp_row.at(s_).index ).cost + distance;
                                } else {
                                    // Pad with insertions
                                    assert(s_ > 0);
                                    tmp_row.at(s_).index = tmp_row.at(s_ - 1).index;
                                    tmp_row.at(s_).cost = tmp_row.at(s_ - 1).cost + 1;
                                }

                                // Increment state index to move along with s_.
                                // This might move closest_index out of the prev_state area
                                closest_index[s]++;
                            }

                            // Forward tmp row
                            update_levenshtein_row(
                                tmp_row,
                                reference_match_index.at(d).at(indices.at(d) - 1),
                                hypothesis_match_index.at(s),
                                matches,
                                new_state.offset.at(s),
                                new_state_end.at(s)
                            );
                        }
                        old_distance = distance;
                        old_closest_index = closest_index;

                        // Copy into state
                        assert(new_state_index[s] == 0);
                        for (unsigned int s_ = 0; s_ < new_state.layout.dimensions.at(s); s_++) {
                            auto _index = new_state.layout.get_index(new_state_index);

                            if (first_update || tmp_row.at(s_).cost < new_state.cost[_index].cost) {
                                new_state.cost[_index].cost = tmp_row.at(s_).cost;
                                new_state.cost[_index].path = std::make_shared<struct Path>(Path{
                                    .previous=prev_state.cost.at(tmp_row.at(s_).index).path,
                                    .speaker=d,
                                    .utterance=indices.at(d) - 1,
                                    .stream=s
                                });
                            }
                            new_state_index[s]++;
                        }

                        new_state_index[s] = 0;
                        if (!new_state.layout.advance_index(new_state_index, s)) break;
                    }
                    first_update = false;
                }
            }

            state.insert({indices, new_state});

            // Increment index by one
            unsigned int d = 0;
            for (; d < num_reference_speakers; d++) {
                assert(indices.at(d) <= end_pointer.at(d));
                if (d == current_speaker_index) continue;
                if (indices.at(d) == end_pointer.at(d)) {
                    indices.at(d) = begin_pointer.at(d);
                } else {
                    indices.at(d)++;
                    break;
                }
            }
            if(d >= num_reference_speakers) break;
        }

        // Stop when all reference utterances have been processed
        bool done = true;
        for (size_t d = 0; d < num_reference_speakers; d++) {
            if (end_pointer.at(d) < reference_utterances.at(d).size()) {
                done = false;
                break;
            }
        }
        if (done) {
            auto final_state = state.at({end_pointer});
            unsigned int cost = final_state.cost.back().cost;

            // Handle any words in the hypothesis that begin after the last reference ended
            for (size_t d = 0; d < num_hypothesis_streams; d++) {
                if (hypothesis.at(d).size() > final_state.offset.at(d) + final_state.layout.dimensions.at(d) - 1) {
                    cost += hypothesis.at(d).size() - final_state.offset.at(d) - final_state.layout.dimensions.at(d) + 1;
                }
            }

            // Get assignment
            // [(reference speaker index, hypothesis stream index)]
            std::vector<std::pair<unsigned int, unsigned int>> assignment;
            Path path = *final_state.cost.back().path;

            while (true) {
                assignment.push_back({path.speaker, path.stream});
                if (!path.previous) { break; }
                path = *path.previous;
            }
            std::reverse(assignment.begin(), assignment.end());
            return std::make_pair(cost, assignment);
        }
    }
}