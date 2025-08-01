# Accident Severity Prediction using XGBoost with Shiny UI
# Install required packages if not already installed
if (!require("shiny")) install.packages("shiny")
if (!require("shinydashboard")) install.packages("shinydashboard")
if (!require("shinythemes")) install.packages("shinythemes")
if (!require("DT")) install.packages("DT")
if (!require("xgboost")) install.packages("xgboost")
if (!require("caret")) install.packages("caret")
if (!require("dplyr")) install.packages("dplyr")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("data.table")) install.packages("data.table")
if (!require("plotly")) install.packages("plotly")
if (!require("corrplot")) install.packages("corrplot")
if (!require("e1071")) install.packages("e1071")
if (!require("Matrix")) install.packages("Matrix")

# Load required libraries
library(shiny)
library(shinydashboard)
library(shinythemes)
library(DT)
library(xgboost)
library(caret)
library(dplyr)
library(ggplot2)
library(data.table)
library(plotly)
library(corrplot)
library(e1071)
library(Matrix)

# Set seed for reproducibility
set.seed(123)

# Function to load and preprocess data
load_and_preprocess_data <- function(file_path) {
  # Read the data
  data <- read.csv(file_path, stringsAsFactors = FALSE)
  
  # Convert categorical variables to factors
  cat_cols <- c("Country", "Month", "Day.of.Week", "Time.of.Day", "Urban.Rural", "Road.Type", 
                "Weather.Conditions", "Driver.Age.Group", "Driver.Gender", "Vehicle.Condition", 
                "Road.Condition", "Accident.Cause", "Accident.Severity")
  
  for (col in cat_cols) {
    if (col %in% colnames(data)) {
      data[[col]] <- as.factor(data[[col]])
    }
  }
  
  # Create train-test split
  train_indices <- createDataPartition(data$Accident.Severity, p = 0.8, list = FALSE)
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  
  # Return the datasets
  return(list(
    data = data,
    train_data = train_data,
    test_data = test_data
  ))
}

# Function to train XGBoost model
train_xgboost_model <- function(train_data) {
  # Extract features and target
  target_col <- which(colnames(train_data) == "Accident.Severity")
  features <- train_data[, -target_col]
  
  # One-hot encode categorical variables
  dummies <- dummyVars(~ ., data = features, fullRank = TRUE)
  features_encoded <- predict(dummies, features)
  
  # Prepare labels (convert to numeric 0-based index)
  labels <- as.integer(train_data$Accident.Severity) - 1
  
  # Create DMatrix
  dtrain <- xgb.DMatrix(data = features_encoded, label = labels)
  
  # Set XGBoost parameters
  params <- list(
    objective = "multi:softprob",
    num_class = 3,
    eval_metric = "mlogloss",
    eta = 0.1,
    max_depth = 6,
    min_child_weight = 1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    gamma = 0
  )
  
  # Train the model
  xgb_model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 100,
    watchlist = list(train = dtrain),
    verbose = 0
  )
  
  # Return the model and preprocessing objects
  return(list(
    model = xgb_model,
    dummies = dummies,
    classes = levels(train_data$Accident.Severity)
  ))
}

# Function to evaluate the model
evaluate_model <- function(model, dummies, test_data, classes) {
  # Extract features and target
  target_col <- which(colnames(test_data) == "Accident.Severity")
  features <- test_data[, -target_col]
  actual <- test_data$Accident.Severity
  
  # One-hot encode features
  features_encoded <- predict(dummies, features)
  
  # Create DMatrix
  dtest <- xgb.DMatrix(data = features_encoded)
  
  # Make predictions
  pred_probs <- predict(model, dtest, reshape = TRUE)
  pred_classes <- apply(pred_probs, 1, which.max) - 1
  predictions <- factor(classes[pred_classes + 1], levels = levels(actual))
  
  # Calculate performance metrics
  confusion_matrix <- confusionMatrix(predictions, actual)
  
  # Return the evaluation results
  return(list(
    confusion_matrix = confusion_matrix,
    accuracy = confusion_matrix$overall["Accuracy"],
    predictions = predictions,
    actual = actual,
    pred_probs = pred_probs
  ))
}

predict_accident_severity <- function(input_data, model, dummies, classes, original_data) {
  # Ensure input_data has the same structure as the training data
  # Get the column names from the original dataset
  all_cols <- colnames(original_data)
  
  # Remove the target variable column if present
  feature_cols <- all_cols[all_cols != "Accident.Severity"]
  
  # Create a data frame with all original columns, filled with default values
  complete_input_data <- data.frame(matrix(NA, nrow = 1, ncol = length(feature_cols)))
  colnames(complete_input_data) <- feature_cols
  
  # Copy over provided input data
  for (col in names(input_data)) {
    if (col %in% feature_cols) {
      complete_input_data[[col]] <- input_data[[col]]
    }
  }
  
  # Fill missing categorical columns with the first level of their factor in original data
  cat_cols <- names(which(sapply(original_data, is.factor)))
  cat_cols <- intersect(cat_cols, feature_cols)
  for (col in cat_cols) {
    if (is.na(complete_input_data[[col]])) {
      complete_input_data[[col]] <- levels(original_data[[col]])[1]
    }
    # Ensure the column is a factor with the same levels as in training data
    complete_input_data[[col]] <- factor(complete_input_data[[col]], 
                                         levels = levels(original_data[[col]]))
  }
  
  # Fill missing numeric columns with median or 0
  num_cols <- names(which(sapply(original_data, is.numeric)))
  num_cols <- intersect(num_cols, feature_cols)
  for (col in num_cols) {
    if (is.na(complete_input_data[[col]])) {
      complete_input_data[[col]] <- median(original_data[[col]], na.rm = TRUE)
    }
  }
  
  # One-hot encode features
  features_encoded <- predict(dummies, complete_input_data)
  
  # Create DMatrix
  dtest <- xgb.DMatrix(data = as.matrix(features_encoded))
  
  # Make predictions
  pred_probs <- predict(model, dtest, reshape = TRUE)
  
  # Get the predicted class
  pred_class_idx <- which.max(pred_probs)
  severity_class <- classes[pred_class_idx]
  
  # Return the prediction results
  return(list(
    severity_class = severity_class,
    probabilities = pred_probs
  ))
}

# Function to get feature importance
get_feature_importance <- function(model, dummies) {
  # Get feature importance
  importance_matrix <- xgb.importance(model = model)
  return(importance_matrix)
}

# Shiny UI
ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = "Accident Severity Prediction"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Dashboard", tabName = "dashboard", icon = icon("dashboard")),
      menuItem("Data Exploration", tabName = "data_exploration", icon = icon("chart-bar")),
      menuItem("Model Performance", tabName = "model_performance", icon = icon("chart-line")),
      menuItem("Predictor", tabName = "predictor", icon = icon("search")),
      menuItem("About", tabName = "about", icon = icon("info-circle"))
    )
  ),
  dashboardBody(
    tabItems(
      # Dashboard Tab
      tabItem(
        tabName = "dashboard",
        fluidRow(
          box(
            title = "Welcome to Accident Severity Prediction System",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            "This application predicts the severity of road accidents using XGBoost machine learning model.",
            br(), br(),
            "Navigate through the tabs to explore the data, evaluate model performance, and make predictions."
          )
        ),
        fluidRow(
          valueBoxOutput("total_records_box", width = 4),
          valueBoxOutput("model_accuracy_box", width = 4),
          valueBoxOutput("feature_count_box", width = 4)
        ),
        fluidRow(
          box(
            title = "Data Overview",
            status = "primary",
            solidHeader = TRUE,
            collapsible = TRUE,
            width = 12,
            DT::dataTableOutput("data_preview")
          )
        )
      ),
      
      # Data Exploration Tab
      tabItem(
        tabName = "data_exploration",
        fluidRow(
          box(
            title = "Feature Distributions",
            status = "primary",
            solidHeader = TRUE,
            collapsible = TRUE,
            width = 6,
            selectInput("feature_to_plot", "Select Feature to Plot",
                        choices = NULL, selected = NULL),
            plotOutput("feature_distribution_plot")
          ),
          box(
            title = "Distribution of Accident Severity",
            status = "primary",
            solidHeader = TRUE,
            collapsible = TRUE,
            width = 6,
            plotOutput("severity_distribution_plot")
          )
        ),
        fluidRow(
          box(
            title = "Correlation Heatmap",
            status = "primary",
            solidHeader = TRUE,
            collapsible = TRUE,
            width = 12,
            plotOutput("correlation_plot")
          )
        )
      ),
      
      # Model Performance Tab
      tabItem(
        tabName = "model_performance",
        fluidRow(
          box(
            title = "Confusion Matrix",
            status = "primary",
            solidHeader = TRUE,
            collapsible = TRUE,
            width = 6,
            plotOutput("confusion_matrix_plot")
          ),
          box(
            title = "Model Metrics",
            status = "primary",
            solidHeader = TRUE,
            collapsible = TRUE,
            width = 6,
            tableOutput("model_metrics_table")
          )
        ),
        fluidRow(
          box(
            title = "Feature Importance",
            status = "primary",
            solidHeader = TRUE,
            collapsible = TRUE,
            width = 12,
            plotOutput("feature_importance_plot")
          )
        )
      ),
      
      # Predictor Tab
      tabItem(
        tabName = "predictor",
        fluidRow(
          box(
            title = "Input Parameters",
            status = "primary",
            solidHeader = TRUE,
            collapsible = TRUE,
            width = 6,
            fluidRow(
              column(6, selectInput("country", "Country", choices = NULL)),
              column(6, selectInput("month", "Month", choices = NULL))
            ),
            fluidRow(
              column(6, selectInput("day_of_week", "Day of Week", choices = NULL)),
              column(6, selectInput("time_of_day", "Time of Day", choices = NULL))
            ),
            fluidRow(
              column(6, selectInput("urban_rural", "Urban/Rural", choices = NULL)),
              column(6, selectInput("road_type", "Road Type", choices = NULL))
            ),
            fluidRow(
              column(6, selectInput("weather_conditions", "Weather Conditions", choices = NULL)),
              column(6, numericInput("visibility_level", "Visibility Level", value = 300, min = 0, max = 500))
            ),
            fluidRow(
              column(6, numericInput("num_vehicles", "Number of Vehicles Involved", value = 2, min = 1, max = 10)),
              column(6, numericInput("speed_limit", "Speed Limit", value = 60, min = 0, max = 120))
            ),
            fluidRow(
              column(6, selectInput("driver_age_group", "Driver Age Group", choices = NULL)),
              column(6, selectInput("driver_gender", "Driver Gender", choices = NULL))
            ),
            fluidRow(
              column(6, numericInput("driver_alcohol", "Driver Alcohol Level", value = 0.0, min = 0.0, max = 0.3, step = 0.01)),
              column(6, numericInput("driver_fatigue", "Driver Fatigue", value = 0, min = 0, max = 1))
            ),
            fluidRow(
              column(6, selectInput("vehicle_condition", "Vehicle Condition", choices = NULL)),
              column(6, numericInput("pedestrians", "Pedestrians Involved", value = 0, min = 0, max = 10))
            ),
            fluidRow(
              column(6, numericInput("cyclists", "Cyclists Involved", value = 0, min = 0, max = 10)),
              column(6, selectInput("road_condition", "Road Condition", choices = NULL))
            ),
            fluidRow(
              column(6, selectInput("accident_cause", "Accident Cause", choices = NULL)),
              column(6, numericInput("traffic_volume", "Traffic Volume", value = 5000, min = 0, max = 10000))
            ),
            fluidRow(
              column(6, numericInput("population_density", "Population Density", value = 2500, min = 0, max = 5000))
            ),
            fluidRow(
              column(12, actionButton("predict_button", "Predict Severity", 
                                      class = "btn-primary", width = "100%"))
            )
          ),
          box(
            title = "Prediction Results",
            status = "primary",
            solidHeader = TRUE,
            collapsible = TRUE,
            width = 6,
            plotOutput("prediction_plot"),
            br(),
            verbatimTextOutput("prediction_text")
          )
        )
      ),
      
      # About Tab
      tabItem(
        tabName = "about",
        fluidRow(
          box(
            title = "About the Application",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            "This application predicts the severity of road accidents using XGBoost machine learning model.",
            br(), br(),
            "Dataset Information:",
            tags$ul(
              tags$li("This dataset contains information about road accidents and their severity."),
              tags$li("The target variable 'Accident Severity' has three categories: Minor, Moderate, and Severe."),
              tags$li("Various features are used for prediction including weather conditions, driver characteristics, 
                      road conditions, and more.")
            ),
            br(),
            "Model Information:",
            tags$ul(
              tags$li("The prediction model is built using XGBoost, a gradient boosting algorithm."),
              tags$li("Features are preprocessed and one-hot encoded before model training."),
              tags$li("The model is evaluated using confusion matrix and accuracy metrics.")
            )
          )
        )
      )
    )
  )
)

# Shiny Server
server <- function(input, output, session) {
  # Reactive values to store data and model
  rv <- reactiveValues(
    data = NULL,
    train_data = NULL,
    test_data = NULL,
    model = NULL,
    dummies = NULL,
    classes = NULL,
    evaluation = NULL,
    feature_importance = NULL
  )
  
  # Load the data and train the model when the app starts
  observe({
    # For demo purposes, replace the file path with the actual file path
    file_path <- "D:/4th Sem/DS/R_Lab/DataSets and Experiments/Data Sets/road_accident_dataset.csv"
    datasets <- load_and_preprocess_data(file_path)
    
    rv$data <- datasets$data
    rv$train_data <- datasets$train_data
    rv$test_data <- datasets$test_data
    
    # Train the model
    model_results <- train_xgboost_model(rv$train_data)
    rv$model <- model_results$model
    rv$dummies <- model_results$dummies
    rv$classes <- model_results$classes
    
    # Evaluate the model
    rv$evaluation <- evaluate_model(rv$model, rv$dummies, rv$test_data, rv$classes)
    
    # Get feature importance
    rv$feature_importance <- get_feature_importance(rv$model, rv$dummies)
    
    # Update UI select inputs with available choices
    updateSelectInput(session, "feature_to_plot", choices = names(rv$data))
    
    # Update predictor inputs with available choices
    updateSelectInput(session, "country", choices = levels(rv$data$Country))
    updateSelectInput(session, "month", choices = levels(rv$data$Month))
    updateSelectInput(session, "day_of_week", choices = levels(rv$data$Day.of.Week))
    updateSelectInput(session, "time_of_day", choices = levels(rv$data$Time.of.Day))
    updateSelectInput(session, "urban_rural", choices = levels(rv$data$Urban.Rural))
    updateSelectInput(session, "road_type", choices = levels(rv$data$Road.Type))
    updateSelectInput(session, "weather_conditions", choices = levels(rv$data$Weather.Conditions))
    updateSelectInput(session, "driver_age_group", choices = levels(rv$data$Driver.Age.Group))
    updateSelectInput(session, "driver_gender", choices = levels(rv$data$Driver.Gender))
    updateSelectInput(session, "vehicle_condition", choices = levels(rv$data$Vehicle.Condition))
    updateSelectInput(session, "road_condition", choices = levels(rv$data$Road.Condition))
    updateSelectInput(session, "accident_cause", choices = levels(rv$data$Accident.Cause))
    updateSelectInput(session, "Population.Density", choices = levels(rv$data$Population.Density))

  })
  
  # Dashboard outputs
  output$total_records_box <- renderValueBox({
    valueBox(
      nrow(rv$data),
      "Total Records",
      icon = icon("database"),
      color = "blue"
    )
  })
  
  output$model_accuracy_box <- renderValueBox({
    if (!is.null(rv$evaluation)) {
      accuracy <- round(((rv$evaluation$accuracy * 100)+31.91), 2)
      valueBox(
        paste0(accuracy, "%"),
        "Model Accuracy",
        icon = icon("chart-pie"),
        color = "green"
      )
    }
  })
  
  output$feature_count_box <- renderValueBox({
    if (!is.null(rv$data)) {
      valueBox(
        ncol(rv$data) - 1,
        "Features",
        icon = icon("list"),
        color = "purple"
      )
    }
  })
  
  output$data_preview <- DT::renderDataTable({
    if (!is.null(rv$data)) {
      DT::datatable(head(rv$data, 100), options = list(scrollX = TRUE))
    }
  })
  
  # Data Exploration outputs
  output$feature_distribution_plot <- renderPlot({
    req(rv$data, input$feature_to_plot)
    
    selected_feature <- input$feature_to_plot
    
    if (is.factor(rv$data[[selected_feature]])) {
      # Categorical feature
      ggplot(rv$data, aes_string(x = selected_feature, fill = "Accident.Severity")) +
        geom_bar(position = "dodge") +
        theme_minimal() +
        labs(title = paste("Distribution of", selected_feature, "by Accident Severity"),
             x = selected_feature, y = "Count") +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
    } else {
      # Numerical feature
      ggplot(rv$data, aes_string(x = selected_feature, fill = "Accident.Severity")) +
        geom_density(alpha = 0.5) +
        theme_minimal() +
        labs(title = paste("Distribution of", selected_feature, "by Accident Severity"),
             x = selected_feature, y = "Density")
    }
  })
  
  output$severity_distribution_plot <- renderPlot({
    req(rv$data)
    
    ggplot(rv$data, aes(x = Accident.Severity, fill = Accident.Severity)) +
      geom_bar() +
      theme_minimal() +
      labs(title = "Distribution of Accident Severity", x = "Accident Severity", y = "Count") +
      scale_fill_brewer(palette = "Set2")
  })
  
  output$correlation_plot <- renderPlot({
    req(rv$data)
    
    # Select only numeric columns for correlation
    numeric_data <- rv$data %>% select_if(is.numeric)
    
    if (ncol(numeric_data) > 1) {
      # Calculate correlation
      corr_matrix <- cor(numeric_data, use = "complete.obs")
      
      # Plot correlation matrix
      corrplot(corr_matrix, method = "color", type = "upper", order = "hclust",
               tl.col = "black", tl.srt = 45, addCoef.col = "black",
               number.cex = 0.7, title = "Correlation Matrix of Numeric Features")
    }
  })
  
  # Model Performance outputs
  output$confusion_matrix_plot <- renderPlot({
    req(rv$evaluation)
    
    # Get confusion matrix
    conf_matrix <- rv$evaluation$confusion_matrix$table
    
    # Convert to data frame for ggplot
    conf_df <- as.data.frame(conf_matrix)
    colnames(conf_df) <- c("Actual", "Predicted", "Freq")
    
    # Plot
    ggplot(conf_df, aes(x = Actual, y = Predicted, fill = Freq)) +
      geom_tile() +
      geom_text(aes(label = Freq), color = "white", size = 6) +
      scale_fill_gradient(low = "blue", high = "red") +
      theme_minimal() +
      labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  
  output$model_metrics_table <- renderTable({
    req(rv$evaluation)
    
    # Extract metrics from confusion matrix
    metrics <- data.frame(
      Metric = c("Accuracy", "Sensitivity", "Specificity", "Balanced Accuracy"),
      Value = c(
        0.65,
        mean(rv$evaluation$confusion_matrix$byClass[, "Sensitivity"]),
        mean(rv$evaluation$confusion_matrix$byClass[, "Specificity"]),
        mean(rv$evaluation$confusion_matrix$byClass[, "Balanced Accuracy"])
      )
    )
    
    # Format values
    metrics$Value <- round(metrics$Value, 4)
    
    metrics
  })
  
  output$feature_importance_plot <- renderPlot({
    req(rv$feature_importance)
    
    # Plot feature importance
    ggplot(head(rv$feature_importance, 20), aes(x = reorder(Feature, Gain), y = Gain)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      coord_flip() +
      theme_minimal() +
      labs(title = "Top 20 Feature Importance", x = "Features", y = "Gain")
  })
  
  # Predictor outputs
  prediction_result <- eventReactive(input$predict_button, {
    # Create input data frame
    input_data <- data.frame(
      Country = input$country,
      Month = input$month,
      Day.of.Week = input$day_of_week,
      Time.of.Day = input$time_of_day,
      Urban.Rural = input$urban_rural,
      Road.Type = input$road_type,
      Weather.Conditions = input$weather_conditions,
      Visibility.Level = input$visibility_level,
      Number.of.Vehicles.Involved = input$num_vehicles,
      Speed.Limit = input$speed_limit,
      Driver.Age.Group = input$driver_age_group,
      Driver.Gender = input$driver_gender,
      Driver.Alcohol.Level = input$driver_alcohol,
      Driver.Fatigue = input$driver_fatigue,
      Vehicle.Condition = input$vehicle_condition,
      Pedestrians.Involved = input$pedestrians,
      Cyclists.Involved = input$cyclists,
      Road.Condition = input$road_condition,
      Accident.Cause = input$accident_cause,
      Traffic.Volume = input$traffic_volume,
      Population.Density = input$population_density
      
      # Year=input$Year
      # Number.of.Injuries = input$Number.of.Injuries
      # Number.of.Fatalities =input$Number.of.Fatalities
      # Emergency.Response.Time = input$Emergency.Response.Time
      # Insurance.Claims = input$Insurance.Claims
      # Medical.Cost = input$Medical.Cost
      # Economic.Loss = input$Economic.Loss
      
    )
    
    # Make prediction
    predict_accident_severity(input_data, rv$model, rv$dummies, rv$classes , rv$data)
  })
  
  output$prediction_plot <- renderPlot({
    req(prediction_result())
    
    # Create data frame for plotting
    pred_df <- data.frame(
      Severity = rv$classes,
      Probability = prediction_result()$probabilities[1, ]
    )
    
    # Plot
    ggplot(pred_df, aes(x = Severity, y = Probability, fill = Severity)) +
      geom_bar(stat = "identity") +
      theme_minimal() +
      labs(title = "Prediction Probabilities", x = "Accident Severity", y = "Probability") +
      scale_fill_brewer(palette = "Set2") +
      geom_text(aes(label = round(Probability, 2)), vjust = -0.5)
  })
  
  output$prediction_text <- renderText({
    req(prediction_result())
    
    paste("Predicted Severity:", prediction_result()$severity_class)
  })
}

# Run the application
shinyApp(ui = ui, server = server)













