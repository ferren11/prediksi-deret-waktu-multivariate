# init.R

my_packages = c("DT", "shinyjs", "plotly", "reticulate", "png")

install_if_missing = function(p) {
  if(p %in% rownames(installed.packages()) == FALSE) {
    install.packages(p)
  }
}

invisible(sapply(my_packages, install_if_missing))

reticulate::install_miniconda()
reticulate::conda_install("r-reticulate", c("keras==2.10.0", "tensorflow-cpu==2.10.0", "pandas==2.2.2", "keras_tuner==1.4.7", "matplotlib==3.8.4", "numpy==1.26.4", "scikit_learn==1.1.3"))

