import numpy as np

# configuration for plausible finger positions
MAX_DISTANCE_MM = 135  # max finger spread distance  # 140
FIRST_FRET_WIDTH = 33  # 36.369
FRET_WIDTH_DECREASE = 1.03  # 1.05946263


def get_all_fret_possibilities(notes, tuning=(64, 59, 55, 50, 45, 40), n_frets=24):
    """ List all possible positions of a note / chord
    
    Parameters
    ----------
    notes: list
        List of pitches
    tuning: tuple of :int:, optional
        tuning of the instrument. Default: standard tuning
    n_frets: int, optional
        number of frets on the fretboard. Default: 24

    Returns
    -------
    list
        list with possible fret positions that result in the given notes.
        list of (list of int)
    """

    solutions = [[-1] * len(tuning)]
    for note in sorted(notes):
        # print(solutions)
        new_solutions = []
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

                        # calculate approximate finger spread width in mm
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
                    # elif c_none > 0:  # append to unlikely solutions
                    #     new_solutions_unlikely.append(new_sol)
                    #     continue
                    # skip if more than 4 different frets are pressed
                    n_different_frets = (np.unique(new_sol[min_idx:max_idx]) > 0).astype('int').sum()
                    if n_different_frets > 4:
                        continue

                    new_solutions.append(new_sol)
        solutions = new_solutions

    return solutions


def get_chord_probability(chord, n_frets=24):
    """ Get probability for a chord
    following characteristics are taken into account:
    - low probability for wide finger spreading (when wider than 2 frets)
    - low probability for empty strings, especially if played strings are on high frets
    - penalty for high frets on thick strings
    - penalty for non-played strings between played strings
    
    Parameters
    ----------
    chord: list of :int:
        note / chord for which to determine the probability
    n_frets: int, optional
        number of frets on the fretboard. Default: 24

    Returns
    -------
    float
        probability / plausibility that this chord is played
    """

    p_fret_diff = 1.0
    penalty = 1.0

    # penalty for high frets on low strings
    checks = min(3, chord.count(0) + chord.count(-1)) + 1
    for i in range(1, checks):
        if chord[len(chord) - i] > 10 + i * 4:  # > 14 for lowest, > 18 for 2nd, > 22 for 3rd string
            penalty *= n_frets / (n_frets + chord[len(chord) - i] - (8 + i * 4))

    played_frets = [fret for fret in chord if fret >= 0]
    if len(played_frets) <= 1:
        return p_fret_diff * penalty

    non_empty_frets = [fret for fret in chord if fret > 0]
    if len(non_empty_frets) > 0:
        fret_diff = max(0, max(non_empty_frets) - min(non_empty_frets) - 2)
        p_fret_diff = 1.0 - min(1.0, fret_diff/10)  # lower possibility for wide spreads

        if len(played_frets) > len(non_empty_frets):  # add penalty if there are empty strings
            penalty = 1.0 - min(non_empty_frets) / n_frets / 2

    min_idx = 0
    while chord[min_idx] == -1 and min_idx < len(chord):
        min_idx += 1
    max_idx = len(chord)
    while chord[max_idx - 1] == -1 and max_idx > 0:
        max_idx -= 1
    c_none = chord[min_idx:max_idx].count(-1)
    # c_empty = new_sol[min_idx:max_idx].count(0)
    for i in range(c_none):  # add penalty for non-played strings in between
        penalty /= 2

    return p_fret_diff * penalty
