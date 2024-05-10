########################################################################
#### Author: Joshua North
#### Project: BayesianSVD
#### Date: 04/16/2024
#### Description: Posterior summaries of length scales
########################################################################

library(tidyverse)
library(ggpubr)


# read data ---------------------------------------------------------------

Udf = read_csv("/Users/JSNorth/Desktop/ULS.csv", name_repair = ~paste0("U_", 1:10)) %>% 
  mutate(n = 1:n())
Vdf = read_csv("/Users/JSNorth/Desktop/VLS.csv", name_repair = ~paste0("V_", 1:10)) %>% 
  mutate(n = 1:n())

# plots -------------------------------------------------------------------


p1 = Udf %>% 
  pivot_longer(-n, names_to = c("U", "Basis"), values_to = "P", names_sep = "_") %>% 
  group_by(Basis) %>% 
  summarise(mean = mean(P), lower = quantile(P, probs = c(0.025)), upper = quantile(P, probs = c(0.975))) %>% 
  mutate(Basis = as.numeric(Basis)) %>% 
  ggplot(., aes(x = Basis)) +
  geom_point(aes(y = mean), size = 3) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.3, linewidth = 1.1) +
  scale_x_continuous(breaks = seq(1,10)) +
  xlab("U") +
  ylab("") +
  theme_bw() +
  theme(axis.text = element_text(size = 16, face = "bold"),
        axis.title = element_text(size = 16, face = "bold"))
  

p2 = Vdf %>% 
  pivot_longer(-n, names_to = c("V", "Basis"), values_to = "P", names_sep = "_") %>% 
  group_by(Basis) %>% 
  summarise(mean = mean(P), lower = quantile(P, probs = c(0.025)), upper = quantile(P, probs = c(0.975))) %>% 
  mutate(Basis = as.numeric(Basis)) %>% 
  ggplot(., aes(x = Basis)) +
  geom_point(aes(y = mean), size = 3) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.3, linewidth = 1.1) +
  scale_x_continuous(breaks = seq(1,10)) +
  xlab("V") +
  ylab("") +
  theme_bw() +
  theme(axis.text = element_text(size = 16, face = "bold"),
        axis.title = element_text(size = 16, face = "bold"))



ggarrange(p1, p2)
ggsave("/Users/JSNorth/.julia/dev/BayesianSVD/figures/lengthScaleEstsV2.pdf", width = 16, height = 5)
# ggsave("/Users/JSNorth/.julia/dev/BayesianSVD/figures/lengthScaleEstsV2.png", width = 16, height = 5)


