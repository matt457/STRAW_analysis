# STRAW_analysis
Analysis of simulations and measurements taken from STRings for Absorption/Attenuation length in Water (STRAW) to determine the optical properties of the deep seawater in the Pacific Ocean. Specifically, we need to determine the absorption, scattering, and attenuation length of the site.

It is planned for a neutrino telescope, P-ONE, to be deployed at the site, so characterization of the optical properties of the water is critical.

# File description
STRAW Analysis Procedure.ipynb is a tutorial on how to run the analysis process from start to finish. It is self contained, so it should rely only on other files in the repo.

clean.py, residual.py, and run.py were written by Akanksha, updated by me. Respectively, the classes have methods that clean the data, plot the photon arrival time residuals, and apply corrections to the overall run. 

P2 Violet.ipynb is a notebook to run clean.py, residual.py, and run.py on the data for violet

Photon MC simulation.ipynb contains the simulations, as well as analysis and comparison to measurements

Attenuation Length.ipynb determines the effective attenuation length from data to help determine absorption and scattering lengths

