import numpy as np

MAX_DISTANCE_MM = 135  # max finger spread distance  # 140
FIRST_FRET_WIDTH = 33  # 36.369
FRET_WIDTH_DECREASE = 1.03  # 1.05946263


def get_all_fret_possibilities(notes, tuning=(64, 59, 55, 50, 45, 40), n_frets=24):
    solutions = [[-1] * len(tuning)]
    for note in sorted(notes):
        # print(solutions)
        new_solutions = []
        new_solutions_unlikely = []
        for sol in solutions:
            for i in range(len(tuning)):
                if sol[i] == -1 and tuning[i] <= note <= tuning[i] + n_frets:
                    new_sol = sol.copy()
                    new_sol[i] = note - tuning[i]

                    min_fret = max_fret = max(new_sol)
                    if max_fret > 0:
                        for ns in new_sol:  # get min fret but without non-played strings!
                            if 0 < ns < min_fret:
                                min_fret = ns

                        # calculate finger spread width in mm (formula for 25.5 inch scale (648mm))
                        distance_mm = 0
                        for f in range(min_fret, max_fret):
                            distance_mm += FIRST_FRET_WIDTH / FRET_WIDTH_DECREASE ** f
                        if distance_mm > MAX_DISTANCE_MM:  # skip if too far apart!
                            continue

                    # check for non-played strings in the middle
                    min_idx = 0
                    while new_sol[min_idx] == -1 and min_idx < len(new_sol):
                        min_idx += 1
                    max_idx = len(new_sol)
                    while new_sol[max_idx-1] == -1 and max_idx > 0:
                        max_idx -= 1
                    c_none = new_sol[min_idx:max_idx].count(-1)
                    # c_empty = new_sol[min_idx:max_idx].count(0)

                    # skip if more than one none-played string or if better solutions are available
                    if c_none > 1:
                        continue
                    elif c_none > 0:  # append to unlikely solutions
                        new_solutions_unlikely.append(new_sol)
                        continue
                    # skip if more than 4 different frets are pressed
                    n_different_frets = (np.unique(new_sol[min_idx:max_idx]) > 0).astype('int').sum()
                    if n_different_frets > 4:
                        continue

                    new_solutions.append(new_sol)
        if len(new_solutions) > 0:
            solutions = new_solutions
        else:
            solutions = new_solutions_unlikely
    return solutions
