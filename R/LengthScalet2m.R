library(gstat)
library(sp)
library(sf)
library(tidyverse)
library(geoR)
library(rgl)
library(spBayes)
library(rgdal)
library(geosphere)


# load data ---------------------------------------------------------------


U = read_csv('/Users/JSNorth/Desktop/lengthScales/Ut2m.csv', col_names = F, name_repair = ~c("lon", "lat", paste0("U", 1:50)))
U = U %>%
  mutate(across(c(U1:U50), ~ (. - mean(.)) / sd(.)))



# ggplot(U, aes(x = lon, y = lat)) +
  # geom_tile(aes(fill = U3)) +
  # scale_fill_gradient2()



coordinates(U) = ~lon+lat
proj4string(U) = "+proj=longlat +ellps=WGS84"

spplot(U, zcol = 'U1', scales = list(draw = TRUE))

TheVariogram=variogram(U1 ~ 1, data=U, cutoff = 1000)
plot(TheVariogram, main = "U1")

# TheVariogramModel <- vgm(psill=1.1, model="Gau", nugget=0.0001, range=1300)
TheVariogramModel <- vgm(psill=1.1, model="Mat", nugget=0.001, range=75, kappa = 3.5)
plot(TheVariogram, model=TheVariogramModel)
FittedModel <- fit.variogram(TheVariogram, model=TheVariogramModel) 
plot(TheVariogram, model=FittedModel)

summary(FittedModel)



Ests = tibble(U = 1:50, sill = NA, range = NA)
for(i in 1:50){
  
  TheVariogram=variogram(as.formula(paste0("U", i, "~1")), data=U, cutoff = 1000)
  # TheVariogram=variogram(as.formula(paste0("U", i, "~lat+lon")), data=U)
  # TheVariogramModel <- vgm(psill=1.6, model="Mat", nugget=0.05, range=125, kappa = 3.5)
  # TheVariogramModel <- vgm(model="Mat", range=150, kappa = 3.5)
  TheVariogramModel <- vgm(model="Gau", range=150)
  FittedModel <- fit.variogram(TheVariogram, model=TheVariogramModel)
  
  Ests$sill[i] = FittedModel$psill
  Ests$range[i] = FittedModel$range
  
  # Ests$sill[i] = FittedModel$psill[2]
  # Ests$range[i] = FittedModel$range[2]
  
}

# hist(Ests$range, breaks = 20)



# all together now

Us = paste0("U", 1:10, collapse = " + ")
TheVariogram=variogram(as.formula(paste0(Us, " ~ 1")), data=U, cutoff = 1000)
# plot(TheVariogram, main = "U1:U10")
TheVariogramModel <- vgm(psill=10, model="Gau", range=175)
# plot(TheVariogram, model=TheVariogramModel)
FittedModel <- fit.variogram(TheVariogram, model=TheVariogramModel) 
# plot(TheVariogram, model=FittedModel)
# summary(FittedModel)
globalRange10 = FittedModel$range




Us = paste0("U", 1:50, collapse = " + ")
TheVariogram=variogram(as.formula(paste0(Us, " ~ 1")), data=U, cutoff = 1000)
plot(TheVariogram, main = "U")


TheVariogramModel <- vgm(psill=50, model="Gau", range=75)
# plot(TheVariogram, model=TheVariogramModel)
FittedModel <- fit.variogram(TheVariogram, model=TheVariogramModel) 
# plot(TheVariogram, model=FittedModel)
# summary(FittedModel)
globalRangeAll = FittedModel$range



p1 = ggplot(Ests, aes(x = U, y = range)) +
  geom_point(size = 3) +
  geom_line(size = 1) +
  geom_hline(yintercept = globalRangeAll, color = "blue", linewidth = 1.2) +
  geom_hline(yintercept = globalRange10, color = "red", linewidth = 1.2) +
  xlab("U Basis Function") +
  ylab("Estimated Length-Scale") +
  scale_y_continuous(breaks = seq(50, 450, 100), limits = c(49, 450)) +
  theme_bw() +
  theme(axis.text = element_text(size = 16, face="bold"),
        axis.title = element_text(size = 16, face="bold"))

# ggsave("../figures/lengthScaleFiguret2mMotivating.png", width = 12, height = 6)






# variogram V -------------------------------------------------------------

V = read_csv('/Users/JSNorth/Desktop/lengthScales/Vt2m.csv', col_names = F, name_repair = ~c("t", paste0("V", 1:50)))
V = V %>%
  mutate(across(c(V1:V50), ~ (. - mean(.)) / sd(.))) %>% 
  mutate(x = 0)




coordinates(V) = ~t+x
# proj4string(V) = "+proj=longlat +ellps=WGS84"

spplot(V, zcol = 'V1', scales = list(draw = TRUE))

TheVariogram=variogram(V1 ~ 1, data=V, cutoff = 7, width = 1)
plot(TheVariogram, main = "V1")

TheVariogramModel <- vgm(psill=1.9, model="Gau", range=3)
# TheVariogramModel <- vgm(psill=0.1, model="Mat", nugget=1, range=5, kappa = 3.5)
# TheVariogramModel <- vgm(psill=0.95, range=12, model="Per", nugget = 0)
plot(TheVariogram, model=TheVariogramModel)
FittedModel <- fit.variogram(TheVariogram, model=TheVariogramModel) 
plot(TheVariogram, model=FittedModel)

summary(FittedModel)



EstsV = tibble(V = 1:50, sill = NA, range = NA)
for(i in 1:50){
  
  TheVariogram=variogram(as.formula(paste0("V", i, "~1")), data=V, cutoff = 7, width = 1)
  # TheVariogram=variogram(as.formula(paste0("V", i, "~lat+lon")), data=V)
  # TheVariogramModel <- vgm(psill=1.6, model="Mat", nugget=0.05, range=125, kappa = 3.5)
  # TheVariogramModel <- vgm(model="Mat", range=3, kappa = 3.5)
  TheVariogramModel <- vgm(model="Gau", range=3, psill = 1.5)
  FittedModel <- fit.variogram(TheVariogram, model=TheVariogramModel)
  
  EstsV$sill[i] = FittedModel$psill
  EstsV$range[i] = FittedModel$range
}




# this does not converge!!!!

# Vs = paste0("V", 1:10, collapse = " + ")
# TheVariogram=variogram(as.formula(paste0(Vs, " ~ 1")), data=V, cutoff = 20, width = 1)
# TheVariogramModel <- vgm(psill=2, model="Gau", range=2, nugget = 7)
# plot(TheVariogram, model=TheVariogramModel)
# 
# FittedModel <- fit.variogram(TheVariogram, model=TheVariogramModel) 
# globalRange10 = FittedModel$range[2]
# 
# 
# 
# 
# Vs = paste0("V", 1:50, collapse = " + ")
# TheVariogram=variogram(as.formula(paste0(Vs, " ~ 1")), data=V, cutoff = 20, width = 1)
# # plot(TheVariogram, main = "V")
# 
# # TheVariogramModel <- vgm(psill=10, model="Gau", range=2, nugget = 37)
# TheVariogramModel <- vgm(psill=15, model="Mat", range=2, nugget = 37)
# # plot(TheVariogram, model=TheVariogramModel)
# FittedModel <- fit.variogram(TheVariogram, model=TheVariogramModel) 
# # plot(TheVariogram, model=FittedModel)
# summary(FittedModel)
# globalRangeAll = FittedModel$range[2]





p2 = ggplot(EstsV, aes(x = V, y = range)) +
  geom_point(size = 3) +
  geom_line(size = 1) +
  # geom_hline(yintercept = globalRangeAll, color = "blue", linewidth = 1.2) +
  # geom_hline(yintercept = globalRange10, color = "red", linewidth = 1.2) +
  xlab("V Basis Function") +
  ylab("") +
  scale_y_continuous(breaks = seq(0, 4, 1), limits = c(0, 4)) +
  theme_bw() +
  theme(axis.text = element_text(size = 16, face="bold"),
        axis.title = element_text(size = 16, face="bold"))



p = cowplot::plot_grid(p1, p2, labels = c("a)", "b)"), label_size = 20, hjust = 0)



ggsave("../figures/lengthScaleFiguret2mMotivating.png", width = 12, height = 6)
# ggsave("/Users/JSNorth/.julia/dev/BayesianSVD/figures/lengthScaleFiguret2mMotivating.png", width = 16, height = 4)









