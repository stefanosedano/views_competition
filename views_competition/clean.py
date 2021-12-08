"""Clean-up"""

from views_competition.cleanup import (
    chadefaux,
    attina,
    baetz,
    benchmark,
    brandt,
    dorazio,
    ettensperger,
    fritz,
    hultman,
    lindholm,
    malone,
    mueller,
    no_change,
    oswald,
    radford,
    randahl,
    vestby,
)


def clean_submissions():
    chadefaux.clean()
    attina.clean()
    baetz.clean()
    benchmark.clean()
    brandt.clean()
    dorazio.clean()
    ettensperger.clean()
    fritz.clean()
    hultman.clean()
    lindholm.clean()
    malone.clean()
    mueller.clean()
    no_change.clean()
    oswald.clean()
    radford.clean()
    randahl.clean()
    vestby.clean()


if __name__ == "__main__":
    clean_submissions()
