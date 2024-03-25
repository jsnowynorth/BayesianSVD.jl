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



ggplot(U, aes(x = lon, y = lat)) +
  geom_tile(aes(fill = U3)) +
  scale_fill_gradient2()



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
  
  # TheVariogram=variogram(as.formula(paste0("U", i, "~1")), data=U)
  TheVariogram=variogram(as.formula(paste0("U", i, "~lat+lon")), data=U)
  # TheVariogramModel <- vgm(psill=1.6, model="Mat", nugget=0.05, range=125, kappa = 3.5)
  TheVariogramModel <- vgm(model="Mat", range=150, kappa = 3.5)
  FittedModel <- fit.variogram(TheVariogram, model=TheVariogramModel)
  
  Ests$sill[i] = FittedModel$psill[2]
  # Ests$range[i] = FittedModel$range[2]
  Ests$range[i] = FittedModel$range
}

hist(Ests$range, breaks = 20)



# all together now
 
# U = read_csv('/Users/JSNorth/Desktop/lengthScales/Ut2m.csv', col_names = F, name_repair = ~c("lon", "lat", paste0("U", 1:50)))
# U = U %>% 
#   pivot_longer(-c(lon, lat), names_to = "basis", values_to = "d") %>% 
#   mutate(d = (d - mean(d)) / sd(d)) %>% 
#   pivot_wider(names_from = basis, values_from = d)
# 
# coordinates(U) = ~lon+lat
# proj4string(U) = "+proj=longlat +ellps=WGS84"


Us = paste0("U", 1:10, collapse = " + ")
TheVariogram=variogram(as.formula(paste0(Us, " ~ 1")), data=U, cutoff = 1000)
plot(TheVariogram, main = "U1:U10")

TheVariogramModel <- vgm(psill=1, model="Mat", nugget=0.001, range=75, kappa = 3.5)
plot(TheVariogram, model=TheVariogramModel)
FittedModel <- fit.variogram(TheVariogram, model=TheVariogramModel) 
plot(TheVariogram, model=FittedModel)

summary(FittedModel)
FittedModel$range[2]

globalRange10 = FittedModel$range[2]




Us = paste0("U", 1:50, collapse = " + ")
TheVariogram=variogram(as.formula(paste0(Us, " ~ 1")), data=U, cutoff = 1000)
plot(TheVariogram, main = "U")


TheVariogramModel <- vgm(psill=50, model="Mat", nugget=0.00001, range=75, kappa = 3.5)
plot(TheVariogram, model=TheVariogramModel)
FittedModel <- fit.variogram(TheVariogram, model=TheVariogramModel) 
plot(TheVariogram, model=FittedModel)

summary(FittedModel)

globalRangeAll = FittedModel$range[2]







# plot of length-scale by basis function ----------------------------------

ggplot(Ests, aes(x = U, y = range)) +
  geom_point(size = 3) +
  geom_line(size = 1) +
  geom_hline(yintercept = globalRangeAll, color = "blue", linewidth = 1.2) +
  geom_hline(yintercept = globalRange10, color = "red", linewidth = 1.2) +
  xlab("Basis Function") +
  ylab("Estimated Length-Scale") +
  scale_y_continuous(breaks = seq(10, 70, 10), limits = c(0, 70)) +
  theme(axis.text = element_text(size = 16, face="bold"),
        axis.title = element_text(size = 20, face="bold"))

# ggsave("/Users/JSNorth/Documents/GitHub/BayesianSpatialBasisFunctions/figures/lengthScaleFiguret2m.png")






# variogram V -------------------------------------------------------------

V = read_csv('/Users/JSNorth/Desktop/lengthScales/Vt2m.csv', col_names = F, name_repair = ~c("t", paste0("V", 1:50)))
V = V %>%
  mutate(across(c(V1:V50), ~ (. - mean(.)) / sd(.))) %>% 
  mutate(x = 0)




coordinates(V) = ~t+x
# proj4string(V) = "+proj=longlat +ellps=WGS84"

spplot(V, zcol = 'V1', scales = list(draw = TRUE))

TheVariogram=variogram(V3 ~ 1, data=V, cutoff = 100)
plot(TheVariogram, main = "V1")

# TheVariogramModel <- vgm(psill=1.1, model="Gau", nugget=0.0001, range=1300)
TheVariogramModel <- vgm(psill=0.1, model="Mat", nugget=1, range=5, kappa = 3.5)
plot(TheVariogram, model=TheVariogramModel)
FittedModel <- fit.variogram(TheVariogram, model=TheVariogramModel) 
plot(TheVariogram, model=FittedModel)

summary(FittedModel)



Ests = tibble(U = 1:50, sill = NA, range = NA)
for(i in 1:50){
  
  # TheVariogram=variogram(as.formula(paste0("U", i, "~1")), data=U)
  TheVariogram=variogram(as.formula(paste0("U", i, "~lat+lon")), data=U)
  # TheVariogramModel <- vgm(psill=1.6, model="Mat", nugget=0.05, range=125, kappa = 3.5)
  TheVariogramModel <- vgm(model="Mat", range=150, kappa = 3.5)
  FittedModel <- fit.variogram(TheVariogram, model=TheVariogramModel)
  
  Ests$sill[i] = FittedModel$psill[2]
  # Ests$range[i] = FittedModel$range[2]
  Ests$range[i] = FittedModel$range
}

hist(Ests$range, breaks = 20)





# check ar parameter for v ------------------------------------------------


V = read_csv('/Users/JSNorth/Desktop/lengthScales/Vt2m.csv', col_names = F, name_repair = ~c("t", paste0("V", 1:50)))
V = V %>%
  mutate(across(c(V1:V50), ~ (. - mean(.)) / sd(.)))

ggplot(V, aes(x = t, y = V2)) +
  geom_line()





EstsV = V %>% 
  summarise(across(V1:V50, ~acf(., plot = F)$acf[2])) %>% 
  pivot_longer(everything(), names_to = "V", values_to = "rho") %>% 
  left_join(V %>% 
              summarise(across(V1:V50, ~pacf(., plot = F)$acf[2])) %>% 
              pivot_longer(everything(), names_to = "V", values_to = "prho")) %>% 
  mutate(basis = 1:n())


ggplot(EstsV, aes(x = basis)) +
  geom_point(size = 3, aes(y = rho), color = "blue") +
  geom_line(size = 1, aes(y = rho), color = "blue") +
  geom_point(size = 3, aes(y = prho), color = "red") +
  geom_line(size = 1, aes(y = prho), color = "red") +
  xlab("Basis Function") +
  ylab("Estimated Correlation Coefficient") +
  scale_y_continuous(breaks = seq(-1, 1, 0.2), limits = c(-1, 1)) +
  theme(axis.text = element_text(size = 16, face="bold"),
        axis.title = element_text(size = 20, face="bold"))



cval = 0.1

EstsV = V %>% 
  summarise(across(V1:V50, ~min(which(abs(acf(., plot = F)$acf) < cval)))) %>% 
  pivot_longer(everything(), names_to = "V", values_to = "rho") %>% 
  left_join(V %>% 
              summarise(across(V1:V50, ~min(which(abs(pacf(., plot = F)$acf) < cval)))) %>% 
              pivot_longer(everything(), names_to = "V", values_to = "prho")) %>% 
  mutate(basis = 1:n())

ggplot(EstsV, aes(x = basis)) +
  geom_point(size = 3, aes(y = rho), color = "blue") +
  geom_line(size = 1, aes(y = rho), color = "blue") +
  geom_point(size = 3, aes(y = prho), color = "red") +
  geom_line(size = 1, aes(y = prho), color = "red") +
  xlab("Basis Function") +
  ylab("Estimated Correlation Coefficient") +
  scale_color_manual(values = c("blue", "red"), labels = c("ACF", "PACF"), name = "") +
  # scale_y_continuous(breaks = seq(-1, 1, 0.2), limits = c(-1, 1)) +
  theme(axis.text = element_text(size = 16, face="bold"),
        axis.title = element_text(size = 20, face="bold"))

