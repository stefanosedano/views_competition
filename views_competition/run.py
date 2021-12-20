"""Top-level runner script"""

import os
import logging
import click
import pickle
import pandas as pd
import numpy as np
from views_competition import (
    config,
    clean,
    collection,
    plot,
    evaluate,
    ensemble,
    bootstrap,
    CLEAN_DIR,
    PICKLE_DIR,
    OUTPUT_DIR,
)
from views_competition.pemdiv import pemdiv_all_t1, pemdiv_all_t2

log = logging.getLogger(__name__)


@click.command()
@click.option("--skip_cleanup", is_flag=True)
@click.option("--skip_collection", is_flag=True)
def run(skip_cleanup, skip_collection):
    """Main clean-up and evaluation runner."""
    if skip_cleanup:
        log.info("Starting without clean-up.")
    else:
        log.info("Starting with clean-up.")
        clean.clean_submissions()

    if skip_collection:
        # Get data and column sets from data/pickled.
        log.info("Starting without submission collection.")
        # TODO: Add optional import function to collection module.
        with open(os.path.join(PICKLE_DIR, "column_sets.pkl"), "rb") as f:
            column_sets = pickle.load(f)
        collection.collect_submissions_from_pickles()
    else:
        # Collect all submission data into our global dfs.
        # Adds to data/pickled.
        column_sets = collection.collect_submissions()

    if config.DO_SCORES:
        log.info("Computing scores for task three...")
        # evaluate.compute_scores(collection.cm_t3, column_sets)
        # evaluate.compute_scores(collection.pgm_t3, column_sets)
        log.info("Computing scores for task two...")
        evaluate.compute_scores(collection.cm_t2, column_sets)
        evaluate.compute_scores(collection.pgm_t2, column_sets)
        log.info("Computing scores for task one, sc...")
        evaluate.compute_scores(collection.cm_t1, column_sets)
        evaluate.compute_scores(collection.pgm_t1, column_sets)
        log.info("Computing scores for task one, ss...")
        evaluate.compute_scores(collection.cm_t1_ss, column_sets)
        evaluate.compute_scores(collection.pgm_t1_ss, column_sets)
        log.info("Writing the 'calib stats' to separate tables.")
        evaluate.write_calibstats(out_path=os.path.join(OUTPUT_DIR, "tables"))

    if config.DO_ENS_T1:
        log.info("Building t1, sc simple avg ensembles.")
        ens_cm_t1 = ensemble.get_simple_sc_ensemble(
            collection.cm_t1, column_sets
        )
        ens_pgm_t1 = ensemble.get_simple_sc_ensemble(
            collection.pgm_t1, column_sets
        )
        log.info("Reshaping t1 ensemble tables to ss columns as well.")
        ens_cm_t1_ss = (
            pd.DataFrame(ens_cm_t1)
            .reset_index()
            .pivot(
                index="country_id",
                columns="month_id",
                values="ensemble",
            )
        )
        ens_cm_t1_ss.columns = [f"ensemble_s{i}" for i in range(2, 8)]
        ens_pgm_t1_ss = (
            pd.DataFrame(ens_pgm_t1)
            .reset_index()
            .pivot(
                index="pg_id",
                columns="month_id",
                values="ensemble",
            )
        )
        ens_pgm_t1_ss.columns = [f"ensemble_s{i}" for i in range(2, 8)]
        log.info("Evaluating t1, sc and ss simple avg ensembles.")
        evaluate.evaluate_t1_sc_ensemble(collection.cm_t1, ens_cm_t1)
        evaluate.evaluate_t1_sc_ensemble(collection.pgm_t1, ens_pgm_t1)
        evaluate.evaluate_t1_ss_ensemble(collection.cm_t1_ss, ens_cm_t1_ss)
        evaluate.evaluate_t1_ss_ensemble(collection.pgm_t1_ss, ens_pgm_t1_ss)
        log.info("Writing t1 sc ensembles to file.")
        ens_cm_t1.to_csv(os.path.join(OUTPUT_DIR, "data", "t1_cm_ens_sc.csv"))
        ens_pgm_t1.to_csv(
            os.path.join(OUTPUT_DIR, "data", "t1_pgm_ens_sc.csv")
        )
        log.info("Writing describe tables to file.")
        # Includes all selected constituent models, and the ensemble.
        # TODO: move this into a function under evaluation.py.
        describe_cm = (
            collection.cm_t1.describe()
            .T[["mean", "std"]]
            .append(
                pd.DataFrame(
                    {
                        "ensemble": ens_cm_t1.describe()
                        .T[["mean", "std"]]
                        .to_dict()
                    }
                ).T
            )
        )
        describe_pgm = (
            collection.pgm_t1.describe()
            .T[["mean", "std"]]
            .append(
                pd.DataFrame(
                    {
                        "ensemble": ens_pgm_t1.describe()
                        .T[["mean", "std"]]
                        .to_dict()
                    }
                ).T
            )
        )
        evaluate.scores_to_tex(
            df=describe_cm.loc[
                [
                    idx
                    for idx in describe_cm.index
                    if idx not in config.DROPS_ENS_T1
                ]
            ],
            out_path=os.path.join(OUTPUT_DIR, "tables", "t1_cm_describe.tex"),
        )
        evaluate.scores_to_tex(
            df=describe_pgm.loc[
                [
                    idx
                    for idx in describe_pgm.index
                    if idx not in config.DROPS_ENS_T1
                ]
            ],
            out_path=os.path.join(OUTPUT_DIR, "tables/t1_pgm_describe.tex"),
        )

    if config.DO_W_ENS_T1:
        if not config.DO_SCORES:
            raise RuntimeError(
                "DO_W_ENS_T1 requires scores. Set config.DO_SCORES to True."
            )
        ensemble.make_ensemble_weights()
        ensemble.write_ensemble_weights(
            ensemble.weights, out_path=os.path.join(OUTPUT_DIR, "tables")
        )
        log.info("Preparing weighted task one ensemble for cm.")
        w_ens_cm_t1 = ensemble.weighted_t1_ensemble(
            collection.cm_t1, column_sets=column_sets
        )
        log.info("Preparing weighted task one ensemble for pgm.")
        w_ens_pgm_t1 = ensemble.weighted_t1_ensemble(
            collection.pgm_t1, column_sets=column_sets
        )
        log.info("Reshaping to ss as well.")
        w_ens_cm_t1_ss = (
            pd.DataFrame(w_ens_cm_t1)
            .reset_index()
            .pivot(
                index="country_id",
                columns="month_id",
                values="w_ensemble",
            )
        )
        w_ens_cm_t1_ss.columns = [f"w_ensemble_s{i}" for i in range(2, 8)]
        w_ens_pgm_t1_ss = (
            pd.DataFrame(w_ens_pgm_t1)
            .reset_index()
            .pivot(
                index="pg_id",
                columns="month_id",
                values="w_ensemble",
            )
        )
        w_ens_pgm_t1_ss.columns = [f"w_ensemble_s{i}" for i in range(2, 8)]
        log.info("Evaluating weighted t1 ensemble.")
        evaluate.evaluate_t1_sc_ensemble(collection.cm_t1, w_ens_cm_t1)
        evaluate.evaluate_t1_sc_ensemble(collection.pgm_t1, w_ens_pgm_t1)
        evaluate.evaluate_t1_ss_ensemble(
            collection.cm_t1_ss, w_ens_cm_t1_ss, weighted=True
        )
        evaluate.evaluate_t1_ss_ensemble(
            collection.pgm_t1_ss, w_ens_pgm_t1_ss, weighted=True
        )
        log.info("Writing weighted t1 ensembles to file.")
        w_ens_cm_t1.to_csv(
            os.path.join(OUTPUT_DIR, "data", "t1_cm_w_ens_sc.csv")
        )
        w_ens_pgm_t1.to_csv(
            os.path.join(OUTPUT_DIR, "data", "t1_pgm_w_ens_sc.csv")
        )

    if config.DO_ENS_T2:
        # Note: adds ensemble to collection data; assumes order.
        log.info("Adding t2 simple avg ensembles.")
        collection.cm_t2 = ensemble.add_simple_ss_ensemble(
            collection.cm_t2, column_sets
        )
        collection.pgm_t2 = ensemble.add_simple_ss_ensemble(
            collection.pgm_t2, column_sets
        )
        log.info("Evaluating t2 ensembles.")
        evaluate.evaluate_t2_ensemble(collection.cm_t2)
        evaluate.evaluate_t2_ensemble(collection.pgm_t2)
        log.info("Writing describe tables to file.")
        evaluate.scores_to_tex(
            df=collection.cm_t2.describe().T[["mean", "std"]],
            out_path=os.path.join(OUTPUT_DIR, "tables", "t2_cm_describe.tex"),
        )
        evaluate.scores_to_tex(
            df=collection.pgm_t2.describe().T[["mean", "std"]],
            out_path=os.path.join(OUTPUT_DIR, "tables", "t2_pgm_describe.tex"),
        )

    if config.DO_DIVERSITY:
        if not config.DO_SCORES:
            raise RuntimeError(
                "DO_DIVERSITY requires scores. Set config.DO_SCORES to True."
            )
        log.info("Preparing diversity stats for the t1, t2 ensembles.")
        evaluate.build_divstats(collection.cm_t1, column_sets)
        evaluate.build_divstats(collection.pgm_t1, column_sets)
        evaluate.build_divstats(collection.cm_t1_ss, column_sets)
        evaluate.build_divstats(collection.pgm_t1_ss, column_sets)
        evaluate.build_divstats(collection.cm_t2, column_sets)
        evaluate.build_divstats(collection.pgm_t2, column_sets)

    if config.DO_ABLATION:
        if not config.DO_SCORES:
            raise RuntimeError(
                "DO_ABLATION requires scores. Set config.DO_SCORES to True."
            )
        log.info("Running ablation studies...")
        evaluate.ablation_study(collection.cm_t1, column_sets)
        evaluate.ablation_study(collection.pgm_t1, column_sets)
        evaluate.ablation_study(collection.cm_t1_ss, column_sets)
        evaluate.ablation_study(collection.pgm_t1_ss, column_sets)
        evaluate.ablation_study(collection.cm_t2, column_sets)
        evaluate.ablation_study(collection.pgm_t2, column_sets)
        log.info("Preparing ablation plots.")
        plot.make_ablation_plots(
            out_path=os.path.join(OUTPUT_DIR, "graphs", "ablation"),
        )

    config.COLUMN_SETS = column_sets
    if config.WRITE_DATA:
        log.info("Writing data to file.")
        # Task one.
        collection.cm_t1.join(ens_cm_t1).join(w_ens_cm_t1).to_csv(
            os.path.join(OUTPUT_DIR, "data", "t1_cm.csv")
        )
        collection.pgm_t1.join(ens_pgm_t1).join(w_ens_pgm_t1).to_csv(
            os.path.join(OUTPUT_DIR, "data", "t1_pgm.csv")
        )
        collection.cm_t1_ss.join(ens_cm_t1_ss).join(w_ens_cm_t1_ss).to_csv(
            os.path.join(OUTPUT_DIR, "data", "t1_cm_ss.csv")
        )
        collection.pgm_t1_ss.join(ens_pgm_t1_ss).join(w_ens_pgm_t1_ss).to_csv(
            os.path.join(OUTPUT_DIR, "data", "t1_pgm_ss.csv")
        )
        # Task two (includes ensemble).
        collection.cm_t2.to_csv(os.path.join(OUTPUT_DIR, "data", "t2_cm.csv"))
        collection.pgm_t2.to_csv(
            os.path.join(OUTPUT_DIR, "data", "t2_pgm.csv")
        )
        # Task three.
        collection.cm_t3.to_csv(os.path.join(OUTPUT_DIR, "data", "t3_cm.csv"))
        collection.pgm_t3.to_csv(
            os.path.join(OUTPUT_DIR, "data", "t3_pgm.csv")
        )

    if config.DO_PEMDIV:
        log.info("Computing pemdiv for t2...")
        pemdiv_all_t2.compute_pemdiv()
        log.info("Computing pemdiv for t1...")
        pemdiv_all_t1.compute_pemdiv()
        log.info("Adding pemdiv to scores from file.")
        evaluate.add_pemdiv()

    # With all scores now collected, write to tables.
    if config.DO_WRITE_SCORES:
        if not (
            config.DO_SCORES and config.DO_DIVERSITY and config.DO_ABLATION and config.DO_PEMDIV
        ):
            raise RuntimeError(
                "DO_WRITE_SCORES requires all scores."
                "Set DO_SCORES, DO_DIVERSITY, DO_ABLATION, DO_PEMDIV to True."
            )
        log.info("Writing t2 and t1 ss scores to pickles and tables.")
        with open(
            os.path.join(OUTPUT_DIR, "tables", "t2_scores.pkl"), "wb"
        ) as f:
            pickle.dump(evaluate.t2_scores, f)
        with open(
            os.path.join(OUTPUT_DIR, "tables", "t1_sc_scores.pkl"), "wb"
        ) as f:
            pickle.dump(evaluate.t1_sc_scores, f)
        with open(
            os.path.join(OUTPUT_DIR, "tables", "t1_ss_scores.pkl"), "wb"
        ) as f:
            pickle.dump(evaluate.t1_ss_scores, f)
        evaluate.write_ss_scores(out_path=os.path.join(OUTPUT_DIR, "tables"))
        log.info("Writing collected t1 sc scores to csv.")
        pd.DataFrame(evaluate.t1_sc_scores["cm"]).T.sort_values(
            by="MSE"
        ).to_csv(os.path.join(OUTPUT_DIR, "tables", "t1_cm_sc_scores.csv"))
        pd.DataFrame(evaluate.t1_sc_scores["pgm"]).T.sort_values(
            by="MSE"
        ).to_csv(os.path.join(OUTPUT_DIR, "tables", "t1_pgm_sc_scores.csv"))
        log.info("Writing collected t1 sc scores to rounded tex.")
        np.round(
            pd.DataFrame(evaluate.t1_sc_scores["cm"]).T.sort_values(by="MSE"),
            3,
        ).to_latex(os.path.join(OUTPUT_DIR, "tables", "t1_cm_sc_scores.tex"))
        np.round(
            pd.DataFrame(evaluate.t1_sc_scores["pgm"]).T.sort_values(by="MSE"),
            3,
        ).to_latex(os.path.join(OUTPUT_DIR, "tables", "t1_pgm_sc_scores.tex"))
        # Also write t1/t2 ensemble tables to file.
        evaluate.write_ensemble_tables(os.path.join(OUTPUT_DIR, "tables"))


    if config.DO_MAPS:
        if not (config.DO_ENS_T1 and config.DO_W_ENS_T1 and config.DO_ENS_T2):
            raise RuntimeError(
                "DO_MAPS requires the ensembles."
                "Set DO_ENS_T1, DO_W_ENS_T1, DO_ENS_T2 to True."
            )
        log.info("Building prediction maps (including ensembles).")
        plot.make_maps(
            collection.cm_t1.join(ens_cm_t1).join(w_ens_cm_t1),
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "maps", "predicted"),
        )
        plot.make_maps(
            collection.pgm_t1.join(ens_pgm_t1).join(w_ens_pgm_t1),
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "maps", "predicted"),
        )
        plot.make_maps(
            collection.cm_t2,
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "maps", "predicted"),
        )
        plot.make_maps(
            collection.pgm_t2,
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "maps", "predicted"),
        )

    if config.DO_ACTUAL_MAPS:
        log.info("Building actuals maps.")
        plot.make_actual_maps(
            collection.cm_t1,
            out_path=os.path.join(OUTPUT_DIR, "maps", "observed"),
        )
        plot.make_actual_maps(
            collection.pgm_t1,
            out_path=os.path.join(OUTPUT_DIR, "maps", "observed"),
        )
        plot.make_actual_maps(
            collection.cm_t2,
            out_path=os.path.join(OUTPUT_DIR, "maps", "observed"),
        )
        plot.make_actual_maps(
            collection.pgm_t2,
            out_path=os.path.join(OUTPUT_DIR, "maps", "observed"),
        )

    if config.DO_ERROR_MAPS:
        if not (config.DO_ENS_T1 and config.DO_W_ENS_T1 and config.DO_ENS_T2):
            raise RuntimeError(
                "DO_ERROR_MAPS requires the ensembles."
                "Set DO_ENS_T1, DO_W_ENS_T1, DO_ENS_T2 to True."
            )
        log.info("Building error maps.")
        plot.make_error_maps(
            collection.cm_t1.join(ens_cm_t1).join(w_ens_cm_t1),
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "maps", "error"),
        )
        plot.make_error_maps(
            collection.pgm_t1.join(ens_pgm_t1).join(w_ens_pgm_t1),
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "maps", "error"),
        )
        plot.make_error_maps(
            collection.cm_t2,
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "maps", "error"),
        )
        plot.make_error_maps(
            collection.pgm_t2,
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "maps", "error"),
        )

    if config.DO_ERROR_PLOTS:
        if not (config.DO_ENS_T1 and config.DO_W_ENS_T1 and config.DO_ENS_T2):
            raise RuntimeError(
                "DO_ERROR_PLOTS requires the ensembles."
                "Set DO_ENS_T1, DO_W_ENS_T1, DO_ENS_T2 to True."
            )
        log.info("Building error plots.")
        plot.make_error_plots(
            collection.cm_t1.join(ens_cm_t1).join(w_ens_cm_t1),
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "graphs", "error"),
        )
        plot.make_error_plots(
            collection.cm_t2,
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "graphs", "error"),
        )
        log.info("Building pgm error plots.")
        plot.make_error_plots_pgm(
            collection.pgm_t1.copy().join(ens_pgm_t1).join(w_ens_pgm_t1),
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "graphs", "error"),
        )
        plot.make_error_plots_pgm(
            collection.pgm_t2.copy(),
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "graphs", "error"),
        )

    if config.DO_LINEPLOTS:
        log.info("Preparing line plots.")
        plot.make_t1_lineplots(
            collection.cm_t1,
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "graphs", "line"),
        )
        plot.make_t1_lineplots(
            collection.pgm_t1,
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "graphs", "line"),
        )
        plot.make_actual_lineplots(
            level="cm", out_path=os.path.join(OUTPUT_DIR, "graphs", "line")
        )
        plot.make_actual_lineplots(
            level="pgm", out_path=os.path.join(OUTPUT_DIR, "graphs", "line")
        )

    if config.DO_CORRPLOTS:
        if not config.DO_SCORES:
            raise RuntimeError(
                "DO_CORRPLOTS requires particular scores."
                "Set DO_SCORES, DO_ABLATION to True."
            )
        log.info("Preparing correlation plots.")
        # Task one.
        plot.make_t1_corrplots(
            collection.cm_t1,
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "graphs/correlation"),
        )
        plot.make_t1_corrplots(
            collection.pgm_t1,
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "graphs", "correlation"),
        )
        # Task two.
        plot.make_t2_corrplots(
            collection.cm_t2,
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "graphs", "correlation"),
        )
        plot.make_t2_corrplots(
            collection.pgm_t2,
            column_sets=column_sets,
            out_path=os.path.join(OUTPUT_DIR, "graphs", "correlation"),
        )

    if config.DO_PCOORD:
        if not (config.DO_SCORES and config.DO_ABLATION):
            raise RuntimeError(
                "DO_PCOORD requires particular scores."
                "Set DO_SCORES, DO_ABLATION to True."
            )
        log.info("Preparing parallel coordinate plots.")
        plot.make_pcoordplots(
            level="cm",
            out_path=os.path.join(OUTPUT_DIR, "graphs", "coordinates"),
        )
        plot.make_pcoordplots(
            level="pgm",
            out_path=os.path.join(OUTPUT_DIR, "graphs", "coordinates"),
        )

    if config.DO_SCATTER:
        if not config.DO_SCORES:
            raise RuntimeError(
                "DO_SCATTER requires particular scores."
                "Set DO_SCORES to True."
            )
        log.info("Preparing scatter plots.")
        plot.make_scatterplots(
            level="cm",
            out_path=os.path.join(OUTPUT_DIR, "graphs", "scatter"),
        )
        plot.make_scatterplots(
            level="pgm",
            out_path=os.path.join(OUTPUT_DIR, "graphs", "scatter"),
        )

    if config.DO_RADAR:
        if not (
            config.DO_SCORES and config.DO_DIVERSITY and config.DO_ABLATION and config.DO_PEMDIV
        ):
            raise RuntimeError(
                "DO_RADAR requires all scores."
                "Set DO_SCORES, DO_DIVERSITY, DO_ABLATION to True."
            )
        log.info("Preparing radar plots.")
        plot.make_radarplots(
            level="cm", out_path=os.path.join(OUTPUT_DIR, "graphs", "radar")
        )
        plot.make_radarplots(
            level="pgm", out_path=os.path.join(OUTPUT_DIR, "graphs", "radar")
        )

    log.info("Finished producing competition output.")

    # Added outputs.
    if config.DO_BOOTSTRAP:
        bootstrap.main()

    if config.DO_MSE_LINES:
        plot.make_mse_lines("cm")
        plot.make_mse_lines("pgm")


if __name__ == "__main__":
    run()
