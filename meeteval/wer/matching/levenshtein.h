#include <vector>
#include <numeric>
#include <stdio.h>
#include <algorithm>
#include <limits>
#include <optional>

unsigned int levenshtein_distance_(
        std::vector<unsigned int> reference,
        std::vector<unsigned int> hypothesis
) {
    // Temporary memory (one row of the levenshtein matrix)
    std::vector<unsigned int> row(hypothesis.size() + 1);

    // Initialize with range
    std::iota(row.begin(), row.end(), 0);

    // Temporary variables
    unsigned int up;    // Value above the current cell (deletion)
    unsigned int diagonal;  // Value diagonal to the current cell (substitution)
    unsigned int left;

    for (auto ref_symbol : reference) {

        auto r = row.begin();
        diagonal = *r;
        (*r)++;
        left = *r;
//        diagonal = row[0]++;  // diagonal = row[0]; row[0] = row[0] + 1;
//        left = row[0];
        r++;
        unsigned int hyp_index = 1;
        for (auto hyp_symbol : hypothesis) {
            up = *r;
//            up = row[hyp_index];
            if (ref_symbol == hyp_symbol) {
                left = diagonal;
            } else {
                left = 1 + std::min(std::min(left, up), diagonal);
            }
            *r = left;
            diagonal = up;
            r++;
//            hyp_index++;
        }
    }
    return row.back();
}

unsigned int levenshtein_distance_custom_cost_(
        std::vector<unsigned int> reference,
        std::vector<unsigned int> hypothesis,
        unsigned int cost_del = 1,
        unsigned int cost_ins = 1,
        unsigned int cost_sub = 1,
        unsigned int cost_cor = 0
) {
    // Temporary memory (one row of the levenshtein matrix)
    std::vector<unsigned int> row(hypothesis.size() + 1);

    // Initialize with range
    row[0] = 0;
    for (unsigned int i = 1; i < row.size(); i++) { row[i] = row[i - 1] + cost_ins; }

    // Temporary variables
    unsigned int up;    // Value above the current cell (deletion)
    unsigned int diagonal;  // Value diagonal to the current cell (substitution)

    for (unsigned int ref_index = 0; ref_index < reference.size(); ref_index++) {
        unsigned int ref_symbol = reference[ref_index];
        diagonal = row[0];
        row[0] += cost_del;

        for (unsigned int hyp_index = 1; hyp_index < hypothesis.size() + 1; hyp_index++) {
            unsigned int hyp_symbol = hypothesis[hyp_index - 1];
            up = row[hyp_index];

            row[hyp_index] = std::min(
                    std::min(
                            row[hyp_index - 1] + cost_ins, // left -> insertion
                            up + cost_del  // up -> deletion
                    ),
                    // diagonal -> correct or substitution
                    diagonal + (hyp_symbol == ref_symbol ? cost_cor : cost_sub)
            );

            diagonal = up;
        }
    }
    return row[row.size() - 1];
}

template<typename T>
bool inline overlaps(const std::pair<T, T> &a, const std::pair<T, T> &b) {
    return a.first < b.second && a.second > b.first;
//    return a.first <= b.second && a.second >= b.first;
}

template<typename T>
unsigned int time_constrained_levenshtein_distance_unoptimized_(
        std::vector<unsigned int> reference,
        std::vector<unsigned int> hypothesis,
        std::vector <std::pair<T, T>> reference_timing,
        std::vector <std::pair<T, T>> hypothesis_timing,
        unsigned int cost_del,
        unsigned int cost_ins,
        unsigned int cost_sub,
        unsigned int cost_cor
) {
    // Temporary memory (one row of the levenshtein matrix)
    std::vector<unsigned int> row(hypothesis.size() + 1);

    // Initialize with range
    row[0] = 0;
    for (unsigned int i = 1; i < row.size(); i++) { row[i] = row[i - 1] + cost_ins; }

    // Temporary variables
    unsigned int up;    // Value above the current cell (deletion)
    unsigned int diagonal;  // Value diagonal to the current cell (substitution)
    unsigned int start_hyp_index = 1;

    for (unsigned int ref_index = 0; ref_index < reference.size(); ref_index++) {
        unsigned int ref_symbol = reference[ref_index];
        std::pair<T, T> ref_interval = reference_timing[ref_index];

        // Update the cells at the border. These are always deletions
        diagonal = row[0];
        row[0] += cost_del;

        for (unsigned int hyp_index = start_hyp_index; hyp_index < hypothesis.size() + 1; hyp_index++) {
            unsigned int hyp_symbol = hypothesis[hyp_index - 1];
            std::pair<T, T> hyp_interval = hypothesis_timing[hyp_index - 1];

            up = row[hyp_index];
            auto ins_or_del = std::min(
                    row[hyp_index - 1] + cost_ins, // left -> insertion
                    up + cost_del  // up -> deletion
            );
            if (overlaps(ref_interval, hyp_interval)) {
                row[hyp_index] = std::min(
                        ins_or_del,
                        // diagonal -> correct or substitution
                        diagonal + (hyp_symbol == ref_symbol ? cost_cor : cost_sub)
                );
            } else {
                row[hyp_index] = ins_or_del;
            }
            diagonal = up;
        }
    }
    return row[row.size() - 1];
}

template<typename T>
unsigned int time_constrained_levenshtein_distance_(
        std::vector<unsigned int> reference,
        std::vector<unsigned int> hypothesis,
        std::vector <std::pair<T, T>> reference_timing,
        std::vector <std::pair<T, T>> hypothesis_timing,
        unsigned int cost_del,
        unsigned int cost_ins,
        unsigned int cost_sub,
        unsigned int cost_cor
) {
    // Temporary memory (one row of the levenshtein matrix)
    std::vector<unsigned int> row(hypothesis.size() + 1);

    // Initialize with range
    row[0] = 0;
    for (unsigned int i = 1; i < row.size(); i++) { row[i] = row[i - 1] + cost_ins; }

    // Temporary variables
    unsigned int up;    // Value above the current cell (deletion)
    unsigned int diagonal;  // Value diagonal to the current cell (substitution)
    unsigned int start_hyp_index = 1;

    // The following variable tracks the maximum seen end time of the reference
    // This is required when the reference intervals don't have increasing end times
    T ref_end_time = reference_timing[0].second;

    for (unsigned int ref_index = 0; ref_index < reference.size(); ref_index++) {
        unsigned int ref_symbol = reference[ref_index];
        std::pair<T, T> ref_interval = reference_timing[ref_index];
        ref_end_time = std::max(ref_interval.second, ref_end_time);

        // Forward to overlapping region. We don't have to use hyp_end_time here because
        // hypothesis_timing[start_hyp_index - 1].second will always be the largest seen end time that overlaps with
        // ref_interval
        for (; start_hyp_index < hypothesis.size() + 1; start_hyp_index++) {
            if (hypothesis_timing[start_hyp_index - 1].second >= ref_interval.first) break;
        }

        // Update the cells at the border. These are always deletions
        diagonal = row[start_hyp_index - 1];
        row[start_hyp_index - 1] += cost_del;

        // Standard levenshtein update with overlap check
        unsigned int hyp_index = start_hyp_index;
        for (; hyp_index < hypothesis.size() + 1; hyp_index++) {
            unsigned int hyp_symbol = hypothesis[hyp_index - 1];
            std::pair<T, T> hyp_interval = hypothesis_timing[hyp_index - 1];

            // We ran outside the overlapping region. We don't need to do the normal levenshtein update here.
            // Below this loop is the update for the insertions in the following cells
            // The begin times (*.first) are increasing, so we only only have to check the begin time of our current
            // hypothesis against the largest seen reference end time
            if (hyp_interval.first > ref_end_time) break;

            // This is the standard levenshtein update but we only allow substitutions when the segments overlap
            up = row[hyp_index];
            auto ins_or_del = std::min(
                    row[hyp_index - 1] + cost_ins, // left -> insertion
                    up + cost_del  // up -> deletion
            );
            if (overlaps(ref_interval, hyp_interval)) {
                row[hyp_index] = std::min(
                        ins_or_del,
                        // diagonal -> correct or substitution
                        diagonal + (hyp_symbol == ref_symbol ? cost_cor : cost_sub)
                );
            } else {
                row[hyp_index] = ins_or_del;
            }
            diagonal = up;
        }

        // Forward insertions for entries that overlap in the next row
        // It is here again enough to check first
        if (ref_index < reference_timing.size() - 1) {
            for (; hyp_index < hypothesis.size() + 1; hyp_index++) {
                row[hyp_index] = row[hyp_index - 1] + cost_ins;
                if (hypothesis_timing[hyp_index - 1].first > reference_timing[ref_index + 1].second) break;
            }
        } else {
            // We reached the end. There are no following rows which we can respect while forwarding. There can only be
            // insertions until the end
            return row[hyp_index - 1] + cost_ins * (hypothesis.size() - hyp_index + 1);
        }
    }
    return row[row.size() - 1];
}

struct LevenshteinStatistics {
    unsigned int insertions;
    unsigned int deletions;
    unsigned int substitutions;
    unsigned int correct;
    unsigned int total;
    std::vector<std::pair<unsigned int, unsigned int>> alignment;
};
const unsigned int ALIGNMENT_EPS = std::numeric_limits<unsigned int>::max();

template<typename T>
LevenshteinStatistics time_constrained_levenshtein_distance_with_alignment_(
        const std::vector<unsigned int> reference,
        const std::vector<unsigned int> hypothesis,
        const std::vector <std::pair<T, T>> reference_timing,
        const std::vector <std::pair<T, T>> hypothesis_timing,
        const unsigned int cost_del,
        const unsigned int cost_ins,
        const unsigned int cost_sub,
        const unsigned int cost_cor,
        const bool prune
) {
    // Temporary memory (one row of the levenshtein matrix)
    const unsigned int max_value = std::numeric_limits<unsigned int>::max() - std::max(std::max(cost_ins, cost_del), std::max(cost_sub, cost_cor));
    if (max_value <= reference.size() + hypothesis.size())
        throw std::runtime_error("costs too large");
    std::vector<std::vector<unsigned int>> matrix(reference.size() + 1, std::vector<unsigned int>(hypothesis.size() + 1, max_value));

    // Initialize with range
    matrix[0][0] = 0;
    for (unsigned int i = 1; i < matrix[0].size(); i++) { matrix[0][i] = matrix[0][i - 1] + cost_ins; }

    // Temporary variables
    unsigned int start_hyp_index = 1;

    // The following variable tracks the maximum seen end time of the reference
    // This is required when the reference intervals don't have increasing end times
    T ref_end_time = reference_timing[0].second;

    for (unsigned int ref_index = 0; ref_index < reference.size(); ref_index++) {
        auto ref_symbol = reference[ref_index];
        auto &prev_row = matrix[ref_index];
        auto &row = matrix[ref_index + 1];

        std::pair<T, T> ref_interval = reference_timing[ref_index];
        ref_end_time = std::max(ref_interval.second, ref_end_time);

        // Forward to overlapping region. We don't have to use hyp_end_time here because
        // hypothesis_timing[start_hyp_index - 1].second will always be the largest seen end time that overlaps with
        // ref_interval
        if (prune) for (; start_hyp_index < hypothesis.size() + 1; start_hyp_index++) {
            if (hypothesis_timing[start_hyp_index - 1].second > ref_interval.first) break;
        }

        // Update the cells at the border. These are always deletions
        row[start_hyp_index - 1] = prev_row[start_hyp_index - 1] + cost_del;

        // Standard levenshtein update with overlap check
        unsigned int hyp_index = start_hyp_index;
        for (; hyp_index < hypothesis.size() + 1; hyp_index++) {
            unsigned int hyp_symbol = hypothesis[hyp_index - 1];
            std::pair<T, T> hyp_interval = hypothesis_timing[hyp_index - 1];

            // We ran outside the overlapping region. We don't need to do the normal levenshtein update here.
            // Below this loop is the update for the insertions in the following cells
            // The begin times (*.first) are increasing, so we only only have to check the begin time of our current
            // hypothesis against the largest seen reference end time
            if (prune && hyp_interval.first > ref_end_time) break;

            // This is the standard levenshtein update but we only allow substitutions when the segments overlap
            auto ins_or_del = std::min(
                    row[hyp_index - 1] + cost_ins, // left -> insertion
                    prev_row[hyp_index] + cost_del  // up -> deletion
            );
            if (overlaps(ref_interval, hyp_interval)) {
                row[hyp_index] = std::min(
                        ins_or_del,
                        // diagonal -> correct or substitution
                        prev_row[hyp_index - 1] + (hyp_symbol == ref_symbol ? cost_cor : cost_sub)
                );
            } else {
                row[hyp_index] = ins_or_del;
            }
        }

        if (prune) {
            // Forward insertions for entries that overlap in the next row
            if (ref_index < reference_timing.size() - 1) {
                for (; hyp_index < hypothesis.size() + 1; hyp_index++) {
                    row[hyp_index] = row[hyp_index - 1] + cost_ins;
                    if (hypothesis_timing[hyp_index - 1].first > reference_timing[ref_index + 1].second) break;
                }
            } else {
                // We reached the end. There are no following rows which we can respect while forwarding. There can only be
                // insertions until the end
                for (; hyp_index < hypothesis.size() + 1; hyp_index++) row[hyp_index] = row[hyp_index - 1] + cost_ins;
            }
        }
    }

    // Backtracking
    unsigned int hyp_index = hypothesis.size();
    unsigned int ref_index = reference.size();
    LevenshteinStatistics statistics;
    statistics.total = matrix.back().back();
    statistics.insertions = 0;
    statistics.deletions = 0;
    statistics.substitutions = 0;
    statistics.correct = 0;

    while (ref_index > 0 || hyp_index > 0) {
        if (ref_index == 0) {
            // always insertion
            statistics.alignment.push_back(std::make_pair(ALIGNMENT_EPS, --hyp_index));
            statistics.insertions++;
        } else if (hyp_index == 0) {
            // always deletion
            statistics.alignment.push_back(std::make_pair(--ref_index, ALIGNMENT_EPS));
            statistics.deletions++;
        } else {
            unsigned int cost_insertion = matrix[ref_index][hyp_index - 1] + cost_ins;
            unsigned int cost_deletion = matrix[ref_index - 1][hyp_index] + cost_del;

            if (overlaps(hypothesis_timing[hyp_index - 1], reference_timing[ref_index - 1])) {
                unsigned int cost_cor_sub = matrix[ref_index - 1][hyp_index - 1];
                if (reference.at(ref_index - 1) == hypothesis.at(hyp_index - 1)) cost_cor_sub += cost_cor;
                else cost_cor_sub += cost_sub;

                if (cost_cor_sub < std::min(cost_deletion, cost_insertion)) {
                    statistics.alignment.push_back(std::make_pair(--ref_index, --hyp_index));
                    if (reference.at(ref_index) != hypothesis.at(hyp_index)) statistics.substitutions++;
                    else statistics.correct++;
                    continue;
                }
            }

           if (cost_deletion < cost_insertion) {
                // deletion
                statistics.alignment.push_back(std::make_pair(--ref_index, ALIGNMENT_EPS));
                statistics.deletions++;
            } else {
                // insertion
                statistics.alignment.push_back(std::make_pair(ALIGNMENT_EPS, --hyp_index));
                statistics.insertions++;
            }
        }
    }

    // Reverse alignment to get correct temporal order
    std::reverse(statistics.alignment.begin(), statistics.alignment.end());

    return statistics;
}

template<typename T>
unsigned int time_constrained_levenshtein_distance_v2_(
        const std::vector<unsigned int> reference,
        const std::vector<unsigned int> hypothesis,
        const std::vector <std::pair<T, T>> reference_timing,
        const std::vector <std::pair<T, T>> hypothesis_timing,
        const unsigned int cost_del,
        const unsigned int cost_ins,
        const unsigned int cost_sub,
        const unsigned int cost_cor
) {
    const unsigned int cost_insdel = cost_ins + cost_del;

    // Temporary memory (one row of the levenshtein matrix)
    std::vector<unsigned int> row(hypothesis.size() + 1, std::numeric_limits<unsigned int>::max());
    row[0] = 0;

    // The following variable tracks the maximum seen end time of the reference
    // This is required when the reference intervals don't have increasing end times
    T ref_end_time = 0;

    unsigned int hyp_start = 0;
    unsigned int hyp_index = 0;

    for (unsigned int ref_index = 0; ref_index < reference.size(); ref_index++) {
        unsigned int ref_symbol = reference[ref_index];
        std::pair<T, T> ref_interval = reference_timing[ref_index];
        ref_end_time = std::max(ref_interval.second, ref_end_time);

        unsigned int diagonal = row[hyp_start];
        row[hyp_start] += cost_del;
        bool allow_shift = true;

        hyp_index = hyp_start;
        for (; hyp_index < hypothesis.size(); hyp_index++) {
            unsigned int hyp_symbol = hypothesis[hyp_index];
            std::pair<T, T> hyp_interval = hypothesis_timing[hyp_index];

            if (allow_shift) {
                if (ref_interval.first > hyp_interval.second) ++hyp_start;
                else allow_shift = false;
            }

            // This happens when row[i+1] is uninitialized
            auto up = std::min(row[hyp_index + 1], diagonal + cost_ins);

            // TODO: do we need the full overlaps check?
             row[hyp_index + 1] = std::min(
                 std::min(
                    row[hyp_index] + cost_ins, // left -> insertion
                    up + cost_del  // up -> deletion
                ),
                diagonal + (overlaps(ref_interval, hyp_interval)
                    ? (hyp_symbol == ref_symbol ? cost_cor : cost_sub)
                    : cost_insdel)
            );

            diagonal = up;

            if (ref_end_time < hyp_interval.first) break;
        }

    }
    for (; hyp_index < hypothesis.size(); hyp_index++) row[hyp_index + 1] = row[hyp_index] + cost_ins;

    return row.back();
}
