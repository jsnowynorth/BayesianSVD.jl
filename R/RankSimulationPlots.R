library(tidyverse)
library(gridExtra)
library(scales)
library(RColorBrewer)

# brewer.pal(5, "Set1")
# load in data ------------------------------------------------------------
# folderfiles <- list.files(path = "/Users/JSNorth/Documents/GitHub/BayesianSpatialBasisFunctions/results/simulationStudy/",
#                           pattern = "\\.csv$",
#                           full.names = TRUE)

folderfiles <- list.files(path = "/Users/JSNorth/Desktop/simulationResults/",
                          pattern = "\\.csv$",
                          full.names = TRUE)

data_csv <- folderfiles %>% 
  set_names() %>% 
  map_dfr(.f = read_delim,
          delim = ",",
          show_col_types = FALSE,
          .id = "file_name")



# create figure -----------------------------------------------------------


df = data_csv %>% 
  select(-file_name) %>% 
  rename("Coverage Y" = "coverData",
         "Coverage U" = "coverU",
         "Coverage V" = "coverV",
         "RMSE Y" = "RMSEData",
         "RMSE U" = "RMSEU",
         "RMSE V" = "RMSEV",
         "RMSE Y A" = "RMSEDataA",
         "RMSE U A" = "RMSEUA",
         "RMSE V A" = "RMSEVA") %>% 
  pivot_longer(-c(replicate:seed))


df_summary = df %>% 
  group_by(k, SNR, name) %>% 
  summarise(mean = median(value),
            lower = quantile(value, probs = 0.025),
            upper = quantile(value, probs = 0.975),
            min = min(value)) %>% 
  ungroup() %>% 
  mutate(SNR = as.factor(SNR),
         k = as.integer(k),
         name = factor(name, levels = c("Coverage Y", "Coverage U", "Coverage V", "RMSE Y", "RMSE U", "RMSE V", "RMSE Y A", "RMSE U A", "RMSE V A"))) %>% 
  arrange(k, SNR)


df_summary = df_summary %>% 
  filter(!str_detect(name, " A")) %>% 
  mutate(variable = str_split(name, " ", simplify = T)[,1],
         metric = str_split(name, " ", simplify = T)[,2])


df_RMSE = df_summary %>% 
  filter(variable == "RMSE")

df_Coverage = df_summary %>% 
  filter(variable != "RMSE")

collist = c("blue4", "orange4", "green3", "red4", "deeppink2", "yellow1")
collist = brewer.pal(6, "Set1")


# 3 by 6 plots ------------------------------------------------------------



ggplot(df_Coverage %>% 
         mutate(metric = factor(metric, levels = c("Y", "U", "V"))), aes(x = k, y = mean)) +
  geom_line(color = "blue4") +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.3, color = NA, fill = "blue") +
  scale_color_manual(values = collist, labels = c("SNR = 0.1","SNR = 0.5", "SNR = 1", "SNR = 2", "SNR = 5", "SNR = 10"), name = "") +
  scale_fill_manual(values = collist, labels = c("SNR = 0.1","SNR = 0.5", "SNR = 1", "SNR = 2", "SNR = 5", "SNR = 10"), name = "") +
  scale_x_continuous("Number of estimated basis functions (Truth: k=5)", breaks = pretty_breaks(n=5)) +
  scale_y_continuous("Coverage", breaks = pretty_breaks(n=5), limits = c(0.35,1)) +
  # ggtitle("Coverage of 95% credible intervals") +
  geom_hline(yintercept = 0.95) +
  geom_vline(xintercept = 5) +
  facet_grid(rows = vars(metric), cols = vars(SNR)) +
  theme(axis.title = element_text(size=14),
        title = element_text(size=14),
        legend.position = "none",
        plot.margin = unit(c(10,10,10,10), "cm")) +
  theme_bw(base_rect_size = 1) +
  guides(fill="none", color = "none") +
  theme(axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = 'bold'),
        title = element_text(size = 16, face = 'bold'),
        strip.text = element_text(size = 16),
        legend.text = element_text(size = 14),
        legend.key.height = unit(1, 'cm'))

# ggsave("/Users/JSNorth/.julia/dev/BayesianSVD/figures/RankSimulationFigureCoverage.pdf", width = 10, height = 6, dpi = 600)
# ggsave("figures/simulationFigureCoverage.png", width = 10, height = 6, dpi = 600)
# ggsave(file="figures/simulationFigureCoverage.eps", device=cairo_ps, width = 10, height = 6, dpi = 600)




df_RMSE = df %>% 
  group_by(k, SNR, name) %>% 
  summarise(mean = median(value),
            lower = quantile(value, probs = 0.025),
            upper = quantile(value, probs = 0.975),
            min = min(value)) %>% 
  ungroup() %>% 
  mutate(SNR = as.factor(SNR),
         k = as.integer(k),
         name = factor(name, levels = c("Coverage Y", "Coverage U", "Coverage V", "RMSE Y", "RMSE U", "RMSE V", "RMSE Y A", "RMSE U A", "RMSE V A"))) %>% 
  arrange(k, SNR) %>% 
  mutate(variable = str_split(name, " ", simplify = T)[,1],
         metric = str_split(name, " ", simplify = T)[,2],
         model = str_split(name, " ", simplify = T)[,3]) %>% 
  mutate(model = if_else(model == "A", "A", "B")) %>% 
  filter(variable == "RMSE") %>% 
  mutate(metric = factor(metric, levels = c("Y", "U", "V")))




ggplot() +
  geom_line(data = df_RMSE %>% filter(model == "B"), aes(x = k, y = mean), color = "blue4") +
  geom_ribbon(data = df_RMSE %>% filter(model == "B"), aes(x = k, ymin = lower, ymax = upper), alpha = 0.3, color = NA, fill = "blue") +
  geom_point(data = df_RMSE %>% filter(model == "A"), aes(x = k, y = mean)) +
  geom_errorbar(data = df_RMSE %>% filter(model == "A"), aes(x = k, ymin = lower, ymax = upper), width = 0.25) +
  scale_color_manual(values = collist, labels = c("SNR = 0.1","SNR = 0.5", "SNR = 1", "SNR = 2", "SNR = 5", "SNR = 10"), name = "") +
  scale_fill_manual(values = collist, labels = c("SNR = 0.1","SNR = 0.5", "SNR = 1", "SNR = 2", "SNR = 5", "SNR = 10"), name = "") +
  scale_x_continuous("Number of estimated basis functions (Truth: k=5)", breaks = pretty_breaks(n=5)) +
  # ggtitle("Root mean square error") +
  ylab("RMSE") +
  geom_vline(xintercept = 5) +
  facet_grid(rows = vars(metric), cols = vars(SNR), scales = "free") +
  theme(axis.title = element_text(size=14),
        title = element_text(size=14),
        legend.position = "none",
        plot.margin = unit(c(10,10,10,10), "cm")) +
  theme_bw(base_rect_size = 1) +
  guides(fill="none", color = "none") +
  theme(axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = 'bold'),
        title = element_text(size = 16, face = 'bold'),
        strip.text = element_text(size = 16),
        legend.text = element_text(size = 14),
        legend.key.height = unit(1, 'cm'))

# ggsave("/Users/JSNorth/.julia/dev/BayesianSVD/figures/RankSimulationFigureRMSE.pdf", width = 10, height = 6, dpi = 600)
# ggsave("figures/simulationFigureRMSE.png", width = 10, height = 6, dpi = 600)
# ggsave(file="figures/simulationFigureRMSE.eps", device=cairo_ps, width = 10, height = 6, dpi = 600)




