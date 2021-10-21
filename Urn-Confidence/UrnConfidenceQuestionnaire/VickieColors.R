library(ggplot2) # for plotting graphs
library(tidyverse)
library(data.table)
library(gridExtra) # for ploting graphs
library(lme4) # for linear regression functions
library(plyr) # for collapse-and-mean functions like ddply
library(psych)
library(GPArotation)
library(paran)
library(reshape)
library(polycor)
library(nFactors)
library(R.matlab)
library(reshape)
library(useful)
library(magrittr)

vickiecolorlist <- data.table::fread("~/Downloads/UMAPfig2A_data.csv") %>% .$colorhex %>% unique
vickiecolorlist %>% as.data.frame() %>% set_colnames("colorhex") %>% write_csv("vickiecolorlist.csv")
