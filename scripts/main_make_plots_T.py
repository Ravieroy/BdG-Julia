# from matplotlib.pyplot import plot
from plot_params import *
from bdg_plotting_utils import *
import sys
import logging

logging.basicConfig(
    filename='plots.log',
    filemode='w',
    format='%(asctime)s  - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)

logging.info("Starting the program to make the plots.")
exit_code = 0

if created_collated_files is True:
    try:
        logging.info("Collating data from directory: %s", post_processing_folder)
        collate_data(seed_list, dict_list, results_folder, post_processing_folder)
        logging.info("Data collated successfully")
    except Exception:
        exit_code = 1
        logging.exception("Exception occurred while collating data. Exiting the program.")
        print("Task exited with errors. See plots.log")
        sys.exit(1)
else:
    logging.info(f"create_collated_files = {created_collated_files}")

if update_collated_files is True:
    try:
        logging.info("Updating collated files.")
        state = info_continued_calc["state"] 
        assert state is True, "info_continued_calc[state] is False"
        update_collated_dict(info_continued_calc)
        logging.info("Data updated successfully")
    except Exception:
        exit_code = 1
        logging.exception("Exception occurred while updating data. Exiting the program.")
        print("Task exited with errors. See plots.log")
        sys.exit(1)
else:
    logging.info(f"update_collated_files = {update_collated_files}")

if info_delta_op_plot["skip_calculation"] is False:
    try:
        logging.info("Making probability distribution plots for order parameter")
        plot_probability_distribution(info_delta_op_plot)
        logging.info("P(Δ) Vs Δ plotted successfully in directory: %s", images_folder)
    except Exception:
        logging.exception("Exception occurred while making P(Δ) plot")
        exit_code = 1


if info_avg_n_plot["skip_calculation"] is False:
    try:
        logging.info("Making probability distribution plots for average density")
        plot_probability_distribution(info_avg_n_plot)
        logging.info("P(n) Vs n plotted successfully in directory: %s", images_folder)
    except Exception:
        logging.exception("Exception occurred while making P(n) plot")
        exit_code = 1


if info_gap_plot["skip_calculation"] is False:
    try:
        logging.info("Making plot for energy gap and superconducting order parameter gap")
        plot_gap(info_gap_plot)
        logging.info("Energy gap plotted successfully in directory: %s", images_folder)
    except Exception:
        logging.exception("Exception occurred while making energy gap plot")
        exit_code = 1


if info_dos_plot["skip_calculation"] is False:
    try:
        logging.info("Making DOS plot")
        plot_dos(info_dos_plot)
        logging.info("DOS plotted successfully in directory: %s", images_folder)
    except Exception:
        logging.exception("Exception occurred while making DOS plot")
        exit_code = 1

if info_prob_zero_plot["skip_calculation"] is False:
    try:
        logging.info("Making P(Δ = 0) plot")
        plot_probability_zero_delta(info_prob_zero_plot)
        logging.info("P(Δ = 0) plotted successfully in directory: %s", images_folder)
    except Exception:
        logging.exception("Exception occurred while making P(Δ = 0) plot")
        exit_code = 1

if info_egap_temp_plot["skip_calculation"] is False:
    try:
        logging.info("Making plot for Egap vs T")
        plot_gap_vs_temp(info_egap_temp_plot)
        logging.info("Egap vs T plotted successfully in directory: %s",images_folder)
    except Exception:
        logging.exception("Exception occurred while making energy gap plot")
        exit_code = 1

if info_deltagap_temp_plot["skip_calculation"] is False:
    try:
        logging.info("Making plot for Δop vs T")
        plot_gap_vs_temp(info_deltagap_temp_plot)
        logging.info("Δop vs T plotted successfully in directory: %s", images_folder)
    except Exception:
        logging.exception("Exception occurred while making energy gap plot")
        exit_code = 1


if info_p_uv_dict["skip_calculation"] is False:
    try:
        logging.info("Creating P_uv dict")
        save_p_uv(info_p_uv_dict)
        logging.info("Created P_uv dict successfully in directory: %s", results_folder)
    except Exception:
        logging.exception("Exception occurred while creating P_uv")
        exit_code = 1

if info_stiffness["skip_calculation"] is False:
    try:
        logging.info("Creating stiffness plot")
        plot_stiffness(info_stiffness)
        logging.info("Created stiffness plot successfully in directory: %s", results_folder)
    except Exception:
        logging.exception("Exception occurred while creating stiffness plots")
        exit_code = 1



if exit_code == 0:
    print("Plotting task completed successfully")
else:
    print("Task exited with errors. See plots.log")
