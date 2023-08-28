#include <vector>
#include <numeric>
#include <stdio.h>
#include <algorithm>
#include <cstddef>


struct UpdateState {
    unsigned int cost;
    unsigned int index;
};

void update_levenshtein_row(
        std::vector<UpdateState> &row,
        const std::vector<unsigned int> &reference,
        const std::vector<unsigned int> &hypothesis
) {
    // Temporary variables
    UpdateState up;    // Value above the current cell (deletion)
    UpdateState diagonal;  // Value diagonal to the current cell (substitution)
    UpdateState left;

    for (auto ref_symbol: reference) {
        diagonal = row[0];
        row[0].cost++;
        left = row[0];

        unsigned int hyp_index = 1;
        for (auto hyp_symbol: hypothesis) {
            up = row[hyp_index];
            if (ref_symbol == hyp_symbol) {
                left = diagonal;
            } else {
                if (up.cost < left.cost) left = up;
                if (diagonal.cost < left.cost) left = diagonal;
                left.cost++; // Cost for ins/del/sub = 1
            }
            row[hyp_index] = left;
            diagonal = up;
            hyp_index++;
        }
    }
}

unsigned int levenshtein_distance_(
        const std::vector<unsigned int> reference,
        const std::vector<unsigned int> hypothesis
) {
    // Temporary memory (one row of the levenshtein matrix)
    std::vector<UpdateState> row(hypothesis.size() + 1);

    // Initialize with range
    for (unsigned int i = 0; i < row.size(); ++i) row[i].cost = i;

    update_levenshtein_row(row, reference, hypothesis);

    return row.back().cost;
}

struct Layout {
    std::vector<size_t> strides;
    std::vector<size_t> dimensions;
    size_t total_size;
};

template<typename T>
Layout inline make_layout(const std::vector<std::vector<T>> vec) {
    Layout layout;
    for (auto v : vec) layout.dimensions.push_back(v.size() + 1);
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

struct State {
    unsigned int cost;
    unsigned int active_reference;
    unsigned int active_hypothesis;
    unsigned int hyp_link;
};

std::pair<unsigned int, std::vector<std::pair<unsigned int, unsigned int>>> mimo_matching_(
        std::vector<std::vector<std::vector<unsigned int>>> references, std::vector<std::vector<unsigned int>> hypotheses
) {
    // Get dimensions and layouts for the storage
    Layout ref_layout = make_layout(references);
    Layout hyp_layout = make_layout(hypotheses);

    // Build reference storage grid
    std::vector<std::vector<State>> ref_grid(ref_layout.total_size);

    // Initialize first element - fill with only deletion errors
    // Ever cell's value is one larger than all its smaller non-diagonal
    // neighbors. We can initialize it by finding one such neighbor, which
    // is always given by index minus the largest stride that is smaller
    // than or equal to the current index
    auto state = std::vector<State>(hyp_layout.total_size);
    state[0].cost = 0;
    unsigned int v = 0;
    for (unsigned int i = 1; i < hyp_layout.total_size; i++) {
        // Find the largest stride that is smaller than or equal to i. Must
        // be increasing with i and v can only increase by one
        if (v < hyp_layout.strides.size() - 1 && i == hyp_layout.strides[v + 1]) v++;
        state[i].cost = state[i - hyp_layout.strides[v]].cost + 1;
    }
    ref_grid[0] = state;

    // This vector stores the indices into the reference utterances
    std::vector<unsigned int> reference_indices(references.size(), 0);

    // Main loop
    for (unsigned int reference_utterance_index = 1; reference_utterance_index < ref_layout.total_size; reference_utterance_index++) {
        // Advance the index by one
        for (unsigned int i = 0; i < references.size(); i++) {
            reference_indices[i]++;
            if (reference_indices[i] < ref_layout.dimensions[i]) break;
            reference_indices[i] = 0;
        }

        // Get destination storage
        auto state = std::vector<State>(hyp_layout.total_size);
        auto first_update = true;

        // Loop through all adjacent reference combinations and advance row
        for (unsigned int active_reference_index = 0; active_reference_index < reference_indices.size(); active_reference_index++) {
            if (reference_indices[active_reference_index] == 0) continue;

            // Get the previous state, only based on the current assignment of
            // references
            std::vector<State> &previous_state = ref_grid[reference_utterance_index - ref_layout.strides[active_reference_index]];
            std::vector<unsigned int> &active_reference = references[active_reference_index][reference_indices[active_reference_index] - 1];

            // Forward levenshtein row
            for (unsigned int active_hypothesis_index = 0; active_hypothesis_index < hypotheses.size(); active_hypothesis_index++) {

                // Iterate over all starting points of of levenshtein rows along the active_hypothesis_index dimension
                // These are all numbers i that satisfy i / hyp_layout.strides[active_hypothesis_index] % hyp_layout.dimensions[active_hypothesis_index] == 0
                for (
                        unsigned int _i = 0;
                        _i < hyp_layout.total_size;
                        _i += hyp_layout.strides.size() > active_hypothesis_index + 1 ? hyp_layout.strides[active_hypothesis_index + 1] : hyp_layout.total_size
                ) {
                    for (
                            unsigned int i = _i;
                            i < _i + hyp_layout.strides[active_hypothesis_index];
                            i++
                    ) {
                        // Copy the levenshtein row into the buffer. This is a strided slicing operation
                        std::vector<UpdateState> tmp_row(hyp_layout.dimensions[active_hypothesis_index]);
                        unsigned int k = i;
                        for (unsigned int j = 0; j < hyp_layout.dimensions[active_hypothesis_index]; j++) {
                            State s = previous_state[k];
                            tmp_row[j].cost = s.cost;
                            tmp_row[j].index = k;
                            k += hyp_layout.strides[active_hypothesis_index];
                        }

                        // Apply levenshtein algorithm
                        update_levenshtein_row(tmp_row, active_reference, hypotheses[active_hypothesis_index]);

                        // Update current state. Keep only the min value. Do inverse of the above stride access
                        k = i;
                        for (auto tmp_state : tmp_row) {
                            if (first_update || state[k].cost > tmp_state.cost) {
                                state[k].cost = tmp_state.cost;
                                state[k].active_reference = active_reference_index;
                                state[k].active_hypothesis = active_hypothesis_index;
                                state[k].hyp_link = tmp_state.index;
                            }
                            k += hyp_layout.strides[active_hypothesis_index];
                        }
                    }
                }
                first_update = false;
            }
        }
        ref_grid[reference_utterance_index] = state;
    }

    // Backtracking for assignment
    std::vector<std::pair<unsigned int, unsigned int>> assignment;
    unsigned int reference_utterance_index = ref_layout.total_size - 1;
    unsigned int hypothesis_index = hyp_layout.total_size - 1;

    while (reference_utterance_index != 0) {
        State current_state = ref_grid[reference_utterance_index][hypothesis_index];
        assignment.push_back(std::make_pair(current_state.active_reference, current_state.active_hypothesis));

        // Update "pointer" variables
        hypothesis_index = current_state.hyp_link;
        reference_utterance_index -= ref_layout.strides[current_state.active_reference];
    }
    std::reverse(assignment.begin(), assignment.end());
    return std::make_pair(ref_grid.back().back().cost, assignment);
}