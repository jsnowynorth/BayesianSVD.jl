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


U = read_csv('~/Desktop/Ut2m.csv', col_names = F, name_repair = ~c("lon", "lat", paste0("U", 1:50)))

U = U %>%
  mutate(across(c(U1:U50), ~ (. - mean(.)) / sd(.)))


# U %>% 
#   select(c(lat, lon)) %>% 
#   st_as_sf(coords = c("lon", "lat"), crs = 4326) %>% 
#   st_distance()

locs = U[,1:2]

coordinates(locs) = ~lon+lat
proj4string(locs) = CRS("+proj=longlat +ellps=WGS84")

coords = spTransform(locs, CRS("+proj=utm +zone=51 ellps=WGS84"))

coords = matrix(c(coords$lon, coords$lat), ncol = 2)
coords = coords/1000

# gtf = as.geodata(U, coords.col = 1:2, data.col = 3:52)
# gtf = as.geodata(U, coords.col = 1:2, data.col = 3)



# compute empirical variogram ---------------------------------------------

max.dist = 0.25*max(iDist(coords))
max.dist = 1000
bins = 20

gtf.variog.mat <- variog(coords = coords, data = U$U1, uvec = seq(0, max.dist, length = bins), trend="1st", pairs.min = 30)
gtf.variog.mod <- variog(coords = coords, data = U$U1, uvec = seq(0, max.dist, length = bins), trend="1st", pairs.min = 30, estimator.type = "modulus") 


# par(mfrow=c(2,2))
plot(gtf.variog.mat)
# plot(gtf.variog.mod)


initial_values = expand.grid(seq(0.00001, 0.00005, by = 0.000005), seq(200, 800, by = 100))

fit.mat = variofit(gtf.variog.mat, ini.cov.pars = initial_values, cov.model = 'matern', kappa = 3.5, weights = "equal", minimisation.function = "nls")
# fit.mod = variofit(gtf.variog.mod, ini.cov.pars = c(0.001, 600), cov.model = 'matern', kappa = 3.5)

plot(gtf.variog.mat)
lines(fit.mat)
lines(fit.mod)




initial_values = expand.grid(seq(0.00001, 0.00005, by = 0.000005), seq(200, 800, by = 100))

initial_values = expand.grid(seq(0.01, 0.5, by = 0.05), seq(100, 100, by = 100))

varioModel <- variog(coords = coords, data = U$U3, uvec = seq(0, 1000, length = bins), pairs.min = 30)
varioFit = variofit(varioModel, ini.cov.pars = initial_values, cov.model = 'matern', kappa = 3.5, weights = "equal", minimisation.function = "nls")

plot(varioModel)
lines(varioFit)














# replicate using t2m data ------------------------------------------------

U = read_csv('~/Desktop/Ut2m.csv', col_names = F, name_repair = ~c("lat", "lon", paste0("U", 1:50)))

U = U %>%
  mutate(across(c(U1:U50), ~ (. - mean(.)) / sd(.)))

# Usf = st_as_sf(U, coords = c("lat", "lon"))

coordinates(U) = ~lat+lon
proj4string(U) = "+proj=longlat +ellps=WGS84"

spplot(U, zcol = 'U1', scales = list(draw = TRUE))

TheVariogram=variogram(U1~1, data=U, cutoff = 1000)
plot(TheVariogram, main = "U1")

# TheVariogramModel <- vgm(psill=1.1, model="Gau", nugget=0.0001, range=1300)
TheVariogramModel <- vgm(psill=0.000107, model="Mat", nugget=0.00001, range=150, kappa = 3.5)
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

# plot of length-scale by basis function ----------------------------------

ggplot(Ests, aes(x = U, y = range)) +
  geom_point(size = 3) +
  geom_line(size = 1) +
  xlab("Basis Function") +
  ylab("Estimated Length-Scale") +
  scale_y_continuous(breaks = seq(50, 650, 100), limits = c(0, 250)) +
  theme(axis.text = element_text(size = 16, face="bold"),
        axis.title = element_text(size = 20, face="bold"))

# ggsave("/Users/JSNorth/Documents/GitHub/BayesianSpatialBasisFunctions/figures/lengthScaleFiguret2m.png")





Urange = read_csv("/Users/JSNorth/Desktop/Ut2mRange.csv", col_names = F, name_repair = ~c("mean", "lower", "upper"))

Uvals = cbind(Ests[1:10,], Urange)


ggplot(Uvals, aes(x = U)) +
  geom_point(aes(y = range), size = 3, color = "black") +
  geom_point(aes(y = mean), size = 3, color = "blue") +
  geom_errorbar(aes(ymin = lower, ymax = upper), color = "blue", width = 0.3)
geom_line(size = 1) +
  xlab("Basis Function") +
  ylab("Estimated Length-Scale") +
  scale_y_continuous(breaks = seq(50, 650, 100), limits = c(0, 250)) +
  theme(axis.text = element_text(size = 16, face="bold"),
        axis.title = element_text(size = 20, face="bold"))







# do on posterior estimates -----------------------------------------------

# U = read_csv('~/Desktop/Ut2m.csv', col_names = F, name_repair = ~c("lat", "lon", paste0("U", 1:50)))
Umean = read_csv('~/Desktop/Ut2mMean.csv', col_names = F, name_repair = ~c("lat", "lon", paste0("U", 1:10)))
Ulower = read_csv('~/Desktop/Ut2mLower.csv', col_names = F, name_repair = ~c("lat", "lon", paste0("U", 11:20)))
Uupper = read_csv('~/Desktop/Ut2mUpper.csv', col_names = F, name_repair = ~c("lat", "lon", paste0("U", 21:30)))


U = Umean %>% 
  left_join(Ulower) %>% 
  left_join(Uupper)

U = U %>%
  mutate(across(c(U1:U30), ~ (. - mean(.)) / sd(.)))

# Usf = st_as_sf(U, coords = c("lat", "lon"))

coordinates(U) = ~lat+lon
proj4string(U) = "+proj=longlat +ellps=WGS84"

spplot(U, zcol = 'U3', scales = list(draw = TRUE))

TheVariogram=variogram(U3~lat+lon, data=U, cutoff = 1000)
plot(TheVariogram, main = "U1")

# TheVariogramModel <- vgm(psill=1.1, model="Gau", nugget=0.0001, range=1300)
# TheVariogramModel <- vgm(psill=0.000107, model="Mat", nugget=0.00001, range=150, kappa = 3.5)
TheVariogramModel <- vgm(model="Mat", range=150, kappa = 3.5)
plot(TheVariogram, model=TheVariogramModel)
FittedModel <- fit.variogram(TheVariogram, model=TheVariogramModel)    
plot(TheVariogram, model=FittedModel)

summary(FittedModel)



Ests = tibble(U = 1:30, sill = NA, range = NA)
for(i in 1:30){
  
  TheVariogram=variogram(as.formula(paste0("U", i, "~1")), data=U)
  # TheVariogramModel <- vgm(psill=0.000107, model="Mat", range=125, kappa = 3.5)
  TheVariogramModel <- vgm(model="Mat", range=150, kappa = 3.5)
  FittedModel <- fit.variogram(TheVariogram, model=TheVariogramModel)
  
  Ests$sill[i] = FittedModel$psill[2]
  # Ests$range[i] = FittedModel$range[2]
  Ests$range[i] = FittedModel$range
}



ggplot(Uvals, aes(x = U)) +
  geom_point(aes(y = range), size = 3, color = "black") +
  geom_point(aes(y = mean), size = 3, color = "blue") +
  geom_errorbar(aes(ymin = lower, ymax = upper), color = "blue", width = 0.3)
geom_line(size = 1) +
  xlab("Basis Function") +
  ylab("Estimated Length-Scale") +
  scale_y_continuous(breaks = seq(50, 650, 100), limits = c(0, 250)) +
  theme(axis.text = element_text(size = 16, face="bold"),
        axis.title = element_text(size = 20, face="bold"))



Ests[1:10,]
Ests[11:20,]
Ests[21:30,]





