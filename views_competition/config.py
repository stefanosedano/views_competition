"""Configuration"""

# Runner switches.
DO_SCORES = False
DO_ENS_T1 = False
DO_W_ENS_T1 = False
DO_ENS_T2 = False
DO_DIVERSITY = False
DO_ABLATION = False
DO_WRITE_SCORES = False
WRITE_DATA = False
DO_MAPS = False
DO_ACTUAL_MAPS = False
DO_ERROR_MAPS = False
DO_ERROR_PLOTS = False
DO_LINEPLOTS = False
DO_CORRPLOTS = False
DO_PCOORD = False
DO_SCATTER = False
DO_RADAR = False
DO_BOOTSTRAP = True

# Column name for the actuals.
COL_OBS = "d_ln_ged_best_sb_s{}"
COL_OBS_09 = "d_ln_ged_best_sb_09_s{}"  # Used to evaluate the benchmark.
COL_OBS_T1 = "d_ln_ged_best_sb"

# Team selections.
DROPS_ENS_T1 = [
    "randahl_hhmm_weighted",
    "randahl_hmm_weighted",
    "vestby_rf_fit",
    "no_change",  # t1 column name without "pgm".
    "no_change_pgm",
    "no_change_cm",  # TODO: distinguish team name and colname here.
    "benchmark",  # Limit t1 ensemble to just the contributions.
    "ensemble",
    "w_ensemble",
]

DROPS_ENS_T2 = [
    "randahl_hhmm_weighted",
    "randahl_hmm_weighted",
    "vestby_rf_fit",
    "ensemble",
]

DROPS_DEFAULT = [
    "randahl_hhmm_weighted",
    "randahl_hmm_weighted",
    "vestby_rf_fit",
]

# Line params.
LINE_COUNTRIES = [
    "Egypt",
    "Mozambique",
    "Cameroon",
]

LINE_PGIDS = [
    147944,  # Mali
    174635,  # Close to Sirte, Libya
    130016,  # North Kivu, DRC
    145108,  # Close to Maiduguri, Cameroon
    137921,  # Close to Bangui, CAR
]

REGION_PGIDS = {
    "mozambique_tanzania": [
        114918,
        114919,
        114920,
        114921,
        114922,
        114198,
        114199,
        114200,
        114201,
        114202,
        113478,
        113479,
        113480,
        113481,
        113482,
        112758,
        112759,
        112760,
        112761,
        112762,
        112038,
        112039,
        112040,
        112041,
        112042,
    ],
    "nigeria_cameroon_chad": [
        149426,
        149427,
        149428,
        149429,
        149430,
        148706,
        148707,
        148708,
        148709,
        148710,
        147986,
        147987,
        147988,
        147989,
        147990,
        147266,
        147267,
        147268,
        147269,
        147270,
        146546,
        146547,
        146548,
        146549,
        146550,
    ],
}
