from views_competition.pemdiv import (
	pemdiv_all_t1, pemdiv_all_t1_ss, pemdiv_all_t2
)

def run_pemdiv():
    pemdiv_all_t2.compute_pemdiv()
    pemdiv_all_t1.compute_pemdiv()


if __name__ == "__main__":
    run_pemdiv()
