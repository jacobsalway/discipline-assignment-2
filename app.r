library(shiny)
library(jsonlite)
library(tidyverse)
library(ggplot2)
library(gridExtra)
library(grid)
library(plotROC)
library(plotly)
library(caret)
library(e1071)

hyperparam_results <- fromJSON('hyperparam_results.json')

hyperparam_results$mse <- apply(hyperparam_results, 1, function(x) MLmetrics::MSE(x$pred, x$truth))
hyperparam_results$accuracy <- apply(hyperparam_results, 1, function(x) MLmetrics::Accuracy(round(x$pred), x$truth))
hyperparam_results$precision <- apply(hyperparam_results, 1, function(x) MLmetrics::Precision(x$truth, round(x$pred)))
hyperparam_results$recall <- apply(hyperparam_results, 1, function(x) MLmetrics::Recall(x$truth, round(x$pred)))
hyperparam_results$f1 <- apply(hyperparam_results, 1, function(x) MLmetrics::F1_Score(x$truth, round(x$pred)))
hyperparam_results$auc <- apply(hyperparam_results, 1, function(x) MLmetrics::AUC(x$pred, x$truth))

ui <- shinyUI(
  navbarPage('',
    tabPanel('Hyperparameter choice', 
      titlePanel('Hyperparameter choices'),
      sidebarLayout(
        sidebarPanel(
          sliderInput(inputId='max_depth', label='Max depth:', min=1, max=10, value=6),
          sliderInput(inputId='gamma', label='Gamma:', min=0, max=20, value=0, step=2.5),
          sliderInput(inputId='alpha', label='Alpha:', min=0, max=1, value=0, step=0.1),
        ),
        mainPanel(
          plotOutput(outputId = 'confusionMat'),
          plotOutput(outputId = 'rocCurve')
        )
      )
    ),
    tabPanel('Hyperparameter distribution',
      titlePanel('Evaluation metric'),
      sidebarLayout(
        sidebarPanel(
          selectInput('metric', 'Metric:',
            c('MSE'='mse', 'Accuracy'='accuracy', 'Precision'='precision', 'Recall'='recall', 'F1'='f1', 'AUC'='auc')
          )
        ),
        mainPanel(
          plotlyOutput(outputId = 'hyperparam', height='800px', width='100%')
        )
      )
    )
  )
)

server <- function(input, output) {
  df <- reactive({
    a = hyperparam_results %>% filter(
      max_depth == input$max_depth &
      gamma == input$gamma &
      alpha == input$alpha
    )
    return(a)
  })
  
  truth <- reactive({
    t = df()
    a = unlist(t$truth)
    return(a)
  })
  pred <- reactive({
    t = df()
    a = unlist(t$pred)
    return(a)
  })
  
  output$confusionMat <- renderPlot({
    y_truth = as.factor(truth())
    y_pred = as.factor(round(pred()))
    cm = confusionMatrix(y_truth, y_pred)
    
    cm_d = as.data.frame(cm$table)
    cm_st = round(data.frame(cm$overall),2) %>% rename(Value = cm.overall)
    cm_c = round(data.frame(cm$byClass),2) %>% rename(Value = cm.byClass)
    cm_p = as.data.frame(prop.table(cm$table))
    cm_d$Perc = round(cm_p$Freq*100,2)
    
    cm_d_p <-  ggplot(data = cm_d, aes(x = Prediction , y =  Reference, fill = Freq))+
      geom_tile() +
      geom_text(aes(label = paste("",Freq,",",Perc,"%")), color = 'white', size = 4) +
      theme_light() +
      theme(aspect.ratio=1) +
      labs(title='Confusion matrix') +
      guides(fill=FALSE)
    
    # plotting the stats
    cm_st_p <- tableGrob(cm_st)
    cm_c_p <- tableGrob(cm_c)
    
    # all together
    grid.arrange(cm_d_p, gtable_combine(cm_st_p, cm_c_p, along=1), nrow = 1, ncol = 2)
  })
  
  output$rocCurve <- renderPlot({
    p = ggplot(data.frame(m=pred(), d=truth()), aes(m=m, d=d)) + 
      geom_roc(n.cuts=0) 
    p + geom_abline(intercept=0, slope=1, color='darkgrey', linetype='dashed') +
      annotate("text", x=.9, y=0, label=paste("AUC =", round(calc_auc(p)$AUC, 2))) +
      theme_classic() + 
      labs(title='ROC curve', x='FPR', y='TPR') + 
      theme(aspect.ratio=1)
  })
  
  output$hyperparam <- renderPlotly({
    metric = hyperparam_results[,input$metric]
    plot_ly() %>% add_trace(data=hyperparam_results, x=~max_depth, y=~alpha, z=~gamma, 
            color=~metric, mode='markers',
            marker=list(size=50*(metric - min(metric)) / (max(metric) - min(metric))),
            text=~paste0(
              'Max depth: ', max_depth,
              '<br>Alpha: ', alpha,
              '<br>Gamma: ', gamma,
              paste0('<br>', input$metric, ":"), round(metric,2)),
            hoverinfo='text') %>%
      layout(title='Hyperparameter distribution')
  })
}

shinyApp(ui=ui, server=server)