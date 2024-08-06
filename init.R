# init.R

my_packages = c("DT", "shinyjs", "plotly", "png")

install_if_missing = function(p) {
  if(p %in% rownames(installed.packages()) == FALSE) {
    install.packages(p)
  }
}

invinsible(sapply(my_packages, install_if_missing))