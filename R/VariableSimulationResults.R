library(tidyverse)



# V1 ----------------------------------------------------------------------



# read_csv("/Users/JSNorth/Desktop/VariableSimulation/results/run1.csv")


folderfiles = list.files(path = "/Users/JSNorth/Desktop/VariableSimulation/results/",
                          pattern = "\\.csv$",
                          full.names = TRUE)

df = folderfiles %>% 
  map_dfr(read_csv)


df %>% 
  pivot_longer(-replicate, names_to = "Name", values_to = "Value") %>% 
  group_by(Name) %>% 
  summarise(m = mean(Value), l = quantile(Value, probs = 0.025), u = quantile(Value, probs = 0.975), s = sd(Value))


df %>% 
  pivot_longer(-replicate, names_to = "Name", values_to = "Value") %>% 
  ggplot(., aes(x = Name, y = Value)) +
  geom_violin() +
  ylim(c(0,1))







# V2 ------------------------------------------------------------------



folderfiles = list.files(path = "/Users/JSNorth/Desktop/VariableSimulation/results/",
                         pattern = "\\.csv$",
                         full.names = TRUE)

rmsefiles = folderfiles[folderfiles %>% str_detect("RMSE")]
runfiles = folderfiles[!folderfiles %>% str_detect("RMSE")]

dfRMSE = rmsefiles %>% 
  map_dfr(read_csv)

dfRUN = runfiles %>% 
  map_dfr(read_csv)



dfRMSE %>% 
  pivot_longer(c(RMSE_U_Var:RMSE_V_Group), names_to = c("drop", "Basis", "Type"), values_to = "RMSE", names_sep = "_") %>% 
  select(-drop) %>% 
  ggplot(., aes(x = Basis, y = RMSE, color = Type)) +
  geom_boxplot() +
  facet_wrap(~SNR+basis, ncol = 4, scales = "free")




dfRMSE %>% 
  mutate(U = RMSE_U_Var/RMSE_U_Group,
         V = RMSE_V_Var/RMSE_V_Group) %>% 
  select(-c(RMSE_U_Var:RMSE_V_Group)) %>% 
  mutate(basis = factor(basis, levels = c(1,2,3,4), ordered = T)) %>% 
  pivot_longer(c(U:V), names_to = "Basis", values_to = "RMSE") %>% 
  ggplot(., aes(x = Basis, y = RMSE, fill = basis), color = "black") +
  geom_hline(yintercept = 1) +
  geom_boxplot(outliers = FALSE) +
  scale_fill_manual(name = "Basis Function", labels = c("1", "2", "3", "4"), values = c('#a6cee3','#1f78b4','#b2df8a','#33a02c')) +
  xlab("") +
  ylab("RMSE Ratio") +
  facet_wrap(~SNR, scales = "free", nrow = 1)

ggsave("../figures/VariableSimulationResults.png", width = 16, height = 4)





dfRMSE %>% 
  mutate(U = RMSE_U_Var/RMSE_U_Group,
         V = RMSE_V_Var/RMSE_V_Group) %>%
  select(-c(RMSE_U_Var:RMSE_V_Group)) %>% 
  mutate(basis = factor(basis, levels = c(1,2,3,4), ordered = T)) %>% 
  group_by(SNR, basis) %>% 
  summarise(U_l = quantile(U, probs = 0.025),
            V_l = quantile(V, probs = 0.025),
            U_u = quantile(U, probs = 0.975),
            V_u = quantile(V, probs = 0.975),
            U_m = quantile(U, probs = 0.5),
            V_m = quantile(V, probs = 0.5),
            U_li = quantile(U, probs = 0.25),
            V_li = quantile(V, probs = 0.25),
            U_ui = quantile(U, probs = 0.75),
            V_ui = quantile(V, probs = 0.75)) %>% 
  ungroup() %>% 
  pivot_longer(-c(SNR, basis), names_to = c("Basis", "metric"), values_to = "RMSE", names_sep = "_") %>% 
  pivot_wider(names_from = "metric", values_from = "RMSE") %>% 
  ggplot(., aes(x = Basis, fill = basis), color = "black") +
  geom_hline(yintercept = 1, linewidth = 1.1) +
  geom_boxplot(aes(lower = li, upper = ui, middle = m, ymin = l, ymax = u), 
               stat = "identity", 
               width = 0.5,
               position = position_dodge(0.7)) +
  scale_fill_manual(name = "Basis\nFunction", labels = c("1", "2", "3", "4"), values = c('#a6cee3','#1f78b4','#b2df8a','#33a02c')) +
  xlab("") +
  ylab("RMSE Ratio") +
  coord_cartesian(ylim = c(0,2.5)) +
  facet_wrap(~SNR, scales = "free", nrow = 1,
             labeller = labeller(SNR = ~paste0("SNR = ", .x))) +
  theme(axis.text = element_text(size = 12, face = "bold"),
        axis.title = element_text(size = 14, face = "bold"),
        legend.text = element_text(size = 10, face = "bold"),
        legend.title = element_text(size = 12, face = "bold"),
        strip.text = element_text(face="bold", size=12)) 

# ggsave("../figures/VariableSimulationResults.pdf", width = 16, height = 4)
# ggsave("../figures/VariableSimulationResults.png", width = 16, height = 4)




dfRUN %>% 
  select(-c(RMSE_Y_Var, RMSE_Y_Group)) %>% 
  pivot_longer(-c(replicate:seed), names_to = c("drop", "Basis", "Model"), values_to = "Coverage", names_sep = "_") %>% 
  select(-c(drop)) %>% 
  ggplot(., aes(x = Basis, fill = Model, y = Coverage)) +
  geom_boxplot() +
  geom_hline(yintercept = 0.95) +
  facet_wrap(~SNR)





dfRUN %>% 
  select(-c(RMSE_Y_Var, RMSE_Y_Group)) %>% 
  pivot_longer(-c(replicate:seed), names_to = c("drop", "Basis", "Model"), values_to = "Coverage", names_sep = "_") %>% 
  group_by(SNR, Basis, Model) %>% 
  summarise(l = quantile(Coverage, probs = 0.025),
            l = quantile(Coverage, probs = 0.025),
            u = quantile(Coverage, probs = 0.975),
            u = quantile(Coverage, probs = 0.975),
            m = quantile(Coverage, probs = 0.5),
            m = quantile(Coverage, probs = 0.5),
            li = quantile(Coverage, probs = 0.25),
            li = quantile(Coverage, probs = 0.25),
            ui = quantile(Coverage, probs = 0.75),
            ui = quantile(Coverage, probs = 0.75)) %>% 
  ungroup() %>% 
  ggplot(., aes(x = Basis, fill = Model)) +
  geom_boxplot(aes(lower = li, upper = ui, middle = m, ymin = l, ymax = u), stat = "identity") +
  geom_hline(yintercept = 0.95) +
  facet_wrap(~SNR)


