library(dplyr)
library(stringr)
library(sf)
library(sfdep)
library(sp)
library(spdep)
library(ggplot2)
library(ggthemes)

#---------------------------------------------------------------
# NEIGHBOURHOOD WEIGHTS
#---------------------------------------------------------------
# Import data for municipalities
data_municipalities <- read.csv("_data/data_municipalities.csv")

# Import a shapefile for municipalities
map_municipalities <- st_read("_data/shapefile/map_municipalities.shp")

# Plot the borders of municipalities
plot(st_geometry(map_municipalities))

# Check if municipality codes in the datafile and the map are the same
which(!map_municipalities$mncplty_c %in% data_municipalities$mncplty_c)
which(!data_municipalities$mncplty_c %in% map_municipalities$mncplty_c)

# "0" is added to some municipality codes, so I need to remove it
map_municipalities$mncplty_c <- str_remove(map_municipalities$mncplty_c, "^0")
data_municipalities$municipality_code <- as.character(data_municipalities$municipality_code)

# Again, check if municipality codes in the datafile and the map are the same
which(!map_municipalities$mncplty_c %in% data_municipalities$mncplty_c)
which(!data_municipalities$mncplty_c %in% map_municipalities$mncplty_c)

# Add the data to the map object
map_municipalities_with_data <- 
  map_municipalities %>% 
  left_join(data_municipalities[,c('municipality_code', 'percent_vaccinated')], 
            by = c('mncplty_c' = 'municipality_code'))

# Visualization for a selected variable -- e.g. percent_vaccinated
ggplot(map_municipalities_with_data, 
       aes(fill = percent_vaccinated)) +
  geom_sf() +
  ggthemes::theme_map() +
  scale_fill_continuous()

# Create spatial weight matrix - contiguity (i.e. common border)
municipalities_neighbours <- sfdep::st_contiguity(map_municipalities)

# Check its structure
glimpse(municipalities_neighbours)

# Converting a list of neighbours into a matrix
spatial_weights <- nb2mat(municipalities_neighbours)

# Check if it is row standardized by default
summary(rowSums(spatial_weights))

# Based on that we can create a variable which is a spatial lag of 
# percent vaccinated (i.e. average value based on the neighbours of 
# each municipality)
map_municipalities_with_data$splag_percent_vaccinated <- 
  as.vector(spatial_weights %*% as.matrix(map_municipalities_with_data$percent_vaccinated))

#-------------------------------------------
# HISTORICAL PARTITIONS OF POLAND
#-------------------------------------------
# Import a shapefile for historical partitions of Poland
map_partitions <- st_read("_data/shapefile/map_partitions.shp")

# Plot historical borders between partitions
ggplot() +
  geom_sf(data = map_partitions, 
          aes(fill = partition)) +
  ggthemes::theme_map()
  
# Overlay the two maps
ggplot() +
  geom_sf(data = map_municipalities_with_data, 
          aes(fill = percent_vaccinated)) +
  ggthemes::theme_map() +
  scale_fill_continuous() +
  geom_sf(data = map_partitions, 
          size = 2, 
          color = "red",
          fill = NA)

# Check if partitions can be added
municipalities_with_border <- st_join(map_municipalities, map_partitions)

#-------------------------------------------
# FULL DATA JOIN
#-------------------------------------------
map_municipalities_with_data <- as.data.frame(map_municipalities_with_data)
municipalities_with_border <- as.data.frame(municipalities_with_border)

all_data <- 
  map_municipalities_with_data[,c('mncplty_c', 'mncplty_n', 'splag_percent_vaccinated')] %>% 
  left_join(municipalities_with_border[,c('mncplty_c', 'partition')], 
            by = c('mncplty_c'))

names(all_data)[names(all_data) == 'mncplty_c'] <- 'municipality_code'
names(all_data)[names(all_data) == 'mncplty_n'] <- 'municipality_name'

write.csv(all_data, '_data/spatial_data.csv', row.names=FALSE)

spatial_data <- read.csv('_data/spatial_data.csv')
