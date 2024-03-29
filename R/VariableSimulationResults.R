library(tidyverse)


# read_csv("/Users/JSNorth/Desktop/VariableSimulation/results/run1.csv")


folderfiles = list.files(path = "/Users/JSNorth/Desktop/VariableSimulation/results/",
                          pattern = "\\.csv$",
                          full.names = TRUE)

df = folderfiles %>% 
  map_dfr(read_csv )


df %>% 
  pivot_longer(-replicate, names_to = "Name", values_to = "Value") %>% 
  group_by(Name) %>% 
  summarise(m = mean(Value), l = quantile(Value, probs = 0.025), u = quantile(Value, probs = 0.975), s = sd(Value))


df %>% 
  pivot_longer(-replicate, names_to = "Name", values_to = "Value") %>% 
  ggplot(., aes(x = Name, y = Value)) +
  geom_violin() +
  ylim(c(0,1))
