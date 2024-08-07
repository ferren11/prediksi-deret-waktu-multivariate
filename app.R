# Import R packages needed for the app ======
library(shiny)
library(DT)
library(shinyjs)
library(plotly)
library(png)

# Define Python packages needed for the app =======rm -f ./.git/index.lock
PYTHON_DEPENDENCIES = c("numpy", "pandas", "scikit-learn", "matplotlib", 
                        "tensorflow", "keras", "keras-tuner")

# App virtualenv setup ===========
# virtualenv_dir = Sys.getenv('VIRTUALENV_NAME')
# python_path = Sys.getenv('PYTHON_PATH')
virtualenv_dir = "myenv"
python_path = "app/.heroku/python"

reticulate::virtualenv_create(envname = virtualenv_dir, python = python_path)
reticulate::virtualenv_install(virtualenv_dir, packages = PYTHON_DEPENDENCIES,
                               ignore_installed = FALSE)
reticulate::use_virtualenv(virtualenv_dir, required = T)


# virtualenv_create(envname = "myenv",
#                   packages = PYTHON_DEPENDENCIES)
# Sys.setenv(RETICULATE_PYTHON = "~/.virtualenvs/myenv/Scripts/python.exe")

# use_virtualenv("myenv", required = TRUE)

# Source the Python script ==========
reticulate::source_python("model.py")

# UI ==================
ui <- navbarPage("Prediksi Deret Waktu Multivariat", id = "navbar", position = "fixed-top",
                 tags$style(type="text/css", "body {padding-top: 70px;}"),
                 tabPanel("Home",
                          fluidPage(
                            useShinyjs(),
                            h2("Selamat Datang di Aplikasi Prediksi Deret Waktu Multivariat"),
                            p("Aplikasi ini menggunakan metode ", tags$strong(tags$em("Long Short Term Memory (LSTM)")), " untuk prediksi deret waktu multivariat. Aplikasi ini dirancang untuk membantu pengguna dalam melakukan prediksi beberapa variabel deret waktu multivariat secara bersamaan dengan menggunakan model LSTM sederhana. Pengguna dapat memberikan input data deret waktu multivariat yang dipunya dan dapat dengan bebas mengatur parameter yang digunakan untuk model LSTM yang dibuat. Selain itu, terdapat juga pilihan untuk melakukan penyetelan hyperparameter untuk mendapat model terbaik dengan nilai evaluasi terendah yang dapat dicapai."),
                            br(),
                            h4("Mengapa menggunakan LSTM?"),
                            p("Model LSTM digunakan dikarenakan model ini dapat menyimpan memori jangka panjang dan jangka pendek secara efisien. Hal ini membuatnya sangat cocok untuk analisis deret waktu yang memerlukan pengambilan keputusan berdasarkan data historis."),
                            br(),
                            p("Mari mulai analisis!"),
                            fluidRow(
                              column(1, actionButton("buttonToInputDatatabPanel", "Mulai")),
                              column(2, actionButton("buttonToPanduanTabPanel", "Lihat Panduan Penggunaan"))
                            )
                          )
                 ),
                 tabPanel("Input Data",
                          sidebarLayout(
                            sidebarPanel(
                              fileInput("file", "Unggah File Dataset Anda (in .csv)", accept = ".csv"),
                              downloadButton("download_sample", "Unduh Data Sampel"),
                              headerPanel(""),
                              headerPanel(""),
                              # tags$div(
                              #   style="margin-bottom:50px;",
                              #   downloadButton("download_sample", "Unduh Data Sampel")
                              # ),
                              selectInput("train_size", "Proporsi Data Training dan Testing:", 
                                          c("70:30"=0.7, "80:20"=0.8, "90:10"=0.9)),
                              numericInput("units", "Jumlah Units dalam Layer LSTM:", 100, min = 1, max = 512), 
                              helpText(" (Antara 1 dan 512)"),  # Helper text for units
                              selectInput("dropout", "Proporsi Dropout:", c("0.0"=0, "0.1"=0.1,
                                                                            "0.2"=0.2, "0.3"=0.3,
                                                                            "0.4"=0.4, "0.5"=0.5)),
                              selectInput("learning_rate", "Learning Rate:", c("0.0001"=0.0001, "0.0005"=0.0005, "0.001"=0.001, "0.005"=0.005, "0.01" = 0.01)),
                              numericInput("epochs", "Jumlah Epochs:", 100, min = 10, max = 200),
                              helpText(" (Antara 10 dan 200)"),  # Helper text for epochs
                              fluidRow(
                                column(2, actionButton("run", "Analisis")),
                                column(4, checkboxInput("tune", "Tuning Hyperparameter", FALSE))
                              ),
                              # actionButton("tune", "Tuning Hyperparameter"),
                              verbatimTextOutput("loading_message"),
                              verbatimTextOutput("error_message")
                              # div(id = "loading_message", "Loading...", style = "display:none;")
                            ),
                            mainPanel(
                              DTOutput("data_table")
                            )
                          ),
                          tags$head(tags$style(HTML("#error_message {color: red; font-weight: bold;}")
                                               )
                                    )
                 ),
                 
                 tabPanel("Hasil",
                          fluidPage(
                            # tags$style(HTML(".plotly-graph-div {margin-bottom: 20px;}")),
                            h3("1. Parameter yang Digunakan"),
                            verbatimTextOutput("params_used"),
                            
                            h3("2. Loss pada Model"),
                            p("Model loss merupakan histori pelatihan model. Nilai ", tags$em("Mean Squared Error"), ", atau MSE, loss merupakan default loss yang digunakan. Tujuan utama dari pelatihan model adalah mengurangi (meminimalkan) nilai loss. Nilai loss menunjukkan seberapa baik atau buruk perilaku model tertentu setelah tiap iterasi (epoch) pengoptimalan. Idealnya, diharapkan pengurangan nilai loss setelah setiap, atau beberapa, iterasi. Model yang baik ditandai dengan nilai loss yang bergerak terus menurun dan kemudian stabil pada nilai mendekati nol."),
                            plotOutput("model_loss"),
                            fluidRow(
                              column(3, downloadButton("download_loss", "Unduh Model Loss")),
                              column(3, downloadButton("download_plot", "Unduh Plot"))
                            ),
                            
                            h3("3. Plot Nilai Prediksi vs. Aktual"),
                            p("Grafik yang menunjukkan pergerakan nilai pada variabel hasil prediksi dan nilai variabel aktual pada data test per bulan. Pergerakan nilai prediksi yang mendekati pergerakan aktual menandakan bahwa model yang dibuat sudah baik dalam melakukan peramalan. Anda dapat melihat besar nilai pada setiap poin waktu dengan cara meng-", tags$em("hover"), "kursor pada garis pergerakan, dimana garis biru menandakan nilai aktual dan garis merah menandakan nilai prediksi. Terdapat pengaturan yang dapat dilakukan dalam melihat besar nilai, yaitu: ", 
                              tags$ul(
                                tags$li("icon garis satu: show closest data on hover."),
                                tags$li("icon garis dua: compare data on hover.")
                              )),
                            uiOutput("plots_ui"), # for dynamic plots
                            
                            h3("4. Evaluasi Model"),
                            p("Evaluasi model menunjukkan besar ", tags$em("Root Mean Squared Error"), ", ", tags$em("Mean Absolute Error"), ", dan", tags$em("Mean Absolute Percentage Error"), " dari data train dan test. Nilai RMSE, MAE, dan MAE yang kecil menandakan bahwa model yang dibuat sudah baik dalam melakukan peramalan. Model LSTM dikatakan sudah baik dalam memodelkan peramalan pergerakan nilai ketika nilai evaluasinya kecil dan konsisten (atau serupa) pada data train dan test."),
                            fluidRow(
                              column(6, h4("Evaluasi Data Train"), DTOutput("train_evaluation_table"), downloadButton("download_train_evaluation", "Unduh Tabel Evaluasi")),
                              column(6, h4("Evaluasi Data Test"), DTOutput("test_evaluation_table"), downloadButton("download_test_evaluation", "Unduh Tabel Evaluasi"))
                            ),
                            
                            h3("5. Tabel Perbandingan Nilai Prediksi dan Aktual"),
                            p("Tabel gabungan antara data hasil prediksi dengan data aktual pada data test."),
                            DTOutput("comparison_table"),
                            downloadButton("download_comparison", "Unduh Tabel Perbandingan")
                          )
                 ),
                 
                 tabPanel("Panduan Penggunaan",
                          fluidPage(
                            h2("Tahapan Penggunaan Aplikasi Prediksi Deret Waktu Multivariat"),
                            p("Aplikasi ini dikembangkan untuk mendemonstrasikan prediksi berbasis Long Short Term Memory (LSTM) untuk analisis deret waktu multivariat"),
                            tags$ol(
                              tags$li(tags$strong("Unggah dataset Anda pada tab 'Input Data'. "), "Aplikasi ini hanya menerima data dalam bentuk format .csv. Selain itu, perlu dipastikan bahwa terdapat titel/penamaan setiap variabel yang diletakkan di setiap baris pertama pada format tabel."),
                              tags$li(tags$strong("Atur parameter untuk model LSTM. "), 
                                      "Terdapat beberapa parameter yang dapat diatur, yaitu:",
                                      tags$ul(
                                        tags$li(tags$strong("Proporsi Data Training dan Testing: "),   "merupakan pembagian seberapa data digunakan untuk pelatihan dan pengujian. Umumnya, data deret waktu yang lebih panjang pada pelatihan (lebih banyak data training) akan memberikan kesempatan kepada model LSTM untuk melatih data dan menangkap informasi pola pergerakan dengan lebih baik sehingga disarankan untuk memilih porsi pelatihan yang lebih besar dibanding porsi pengujian."),
                                        tags$li(tags$strong("Jumlah Units dalam Layer LSTM: "), "merupakan jumlah neuron dalam sebuah layer LSTM. Semakin banyak jumlah units, maka model memiliki kapasitas yang lebih besar dalam menangkap pola kompleks dalam data. Namun, hal tersebut dapat mengarah pada overfitting, dimana model terlalu menyesuaikan diri pada data pelatihan dan buruk dalam pengujian. Kebalikannya, model dengan jumlah units lebih sedikit dapat mengakibatkan terjadinya underfitting, dimana model tidak cukup baik dalam menangkap pola dari data."),
                                        tags$li(tags$strong("Proporsi Dropout: "), "merupakan teknik regularisasi yang digunakan untuk mencegah overfitting dengan cara 'mematikan' (drop) sejumlah unit secara acak dalam sebuah layer selama pelatihan. Proporsi dropout yang terlalu tinggi dapat menghambat model dalam belajar dari data. Sementara itu, proporsi dropout yang terlalu rendah meningkatkan risiko overfitting."),
                                        tags$li(tags$strong("Learning Rate: "), "menentukan seberapa besar pembaruan yang dilakukan pada bobot model setiap kali melalui proses pembelajaran. Learning rate yang lebih tinggi (misalnya 0.01) membuat model belajar lebih cepat, tetapi dapat menyebabkan model melewati minimum global dalam fungsi loss dan mengakibatkan kinerja yang buruk. Sementara itu, learning rate yang lebih rendah (misalnya 0.0001) membuat model belajar lebih lambat, membuat model dapat menemukan minimum global dalam fungsi loss, tetapi memakan waktu lebih lama."),
                                        tags$li(tags$strong("Jumlah Epochs: "), "merupakan satu putaran penuh melalui seluruh dataset dalam proses pelatihan model. Jumlah epoch yang lebih banyak memungkinkan model untuk belajar lebih banyak dari data, tetapi dapat menyebabkan overfitting bila model belajar terlalu lama dari data training. Kebalikannya, epochs yang terlalu sedikit dapat membuat model tidak cukup belajar dari data, berimbas pada terjadinya underfitting."),
                                        tags$li(tags$strong("Tune Hyperparameters: "), "merupakan cara untuk mengoptimalkan performa model LSTM. Pengoptimalan dilakukan dengan melibatkan penyesuaian terhadap beberapa hyperparameter secara sistematis untuk menemukan kombinasi parameter terbaik yang meminimalkan metrik galat (RMSE, MAE, dan MAPE). Penyetelan hyperparameter dilakukan pada parameter units, dropouts, dan learning_rate.", tags$li("Anda dapat mencentang checkbox sebelum analisis untuk melakukan pemodelan dengan menggunakan penyetelan hyperparameter."))
                                      )),
                              tags$li(tags$strong("Klik tombol 'Analisis' untuk memulai pelatihan model.")),
                              tags$li(tags$strong("Lihat hasil analisis pada tab 'Hasil'."), "Pada tab 'Hasil', terdapat beberapa bagian yang dapat anda lihat dengan penjelasan sebagai berikut:",
                                      tags$ul(
                                        tags$li(tags$strong("Parameter yang Digunakan: "), "menunjukkan parameter yang digunakan dalam pemodelan LSTM."),
                                        tags$li(tags$strong("Loss pada Model: "), "menunjukkan visualisasi berupa grafik pergerakan nilai loss dari data pelatihan."), 
                                        tags$li(tags$strong("Plot Nilai Prediksi vs. Aktual: "), "menunjukkan visualisasi berupa grafik pergerakan nilai prediksi dibandingkan dengan nilai aktual pada tiap variabel dalam data."),
                                        tags$li(tags$strong("Evaluasi Model: "), "menunjukkan tabel evaluasi model pada data pelatihan dan data pengujian dengan menggunakan metrics ", tags$em("Root Mean Squared Error"), ", ", tags$em("Mean Absolute Error"), ", dan", tags$em("Mean Absolute Percentage Error"), "."),
                                        tags$li(tags$strong("Tabel Perbandingan Nilai Prediksi dan Aktual: "), "menunjukkan tabel berisikan nilai prediksi dan nilai aktual dari data test.")
                                      ))
                            ),
                            p("Apabila terdapat error atau pertanyaan, dapat langsung menghubungi kontak berikut:", 
                              br(),
                              "feliciaferren@gmail.com (Felicia)",
                              br(),
                              "©Felicia, 2024")
                            
                          )
                 )
)


# Server ==============
server <- function(input, output, session) {
  # Lock the "Hasil" tab initially
  shinyjs::disable(selector = 'a[data-value="Hasil"]')
  
  # Hide loading message initially
  shinyjs::hide("loading_message")
  
  # Download sample data
  output$download_sample <- downloadHandler(
    filename = function() { "sample-data.csv" },
    content = function(file) {
      data = read.csv("data-all.csv")
      write.csv(data, file, row.names = FALSE)
    }
  )
  
  # Display input data in DataTable
  observeEvent(input$file, {
    req(input$file)
    
    # Check if the uploaded file is a CSV
    if (tools::file_ext(input$file$name) != "csv") {
      output$error_message <- renderText({
        "Error: Harap mengunggah file dalam format .csv."
      })
      output$data_table <- renderDT({
        datatable(data.frame())
      })
      return(NULL)
    }
    
    # Clear the error message if the file is valid
    output$error_message <- renderText({
      NULL
    })
    
    data <- read.csv(input$file$datapath)
    output$data_table <- renderDT({
      datatable(data)
    })
  })
  
  # Switch to "Input Data" tab when the "Input Data" button is clicked
  observeEvent(input$buttonToInputDatatabPanel, {
    updateNavbarPage(session, "navbar", selected = "Input Data")
  })
  
  # Switch to "Panduan Penggunaan" tab when the "Lihat Panduan Penggunaan" button is clicked
  observeEvent(input$buttonToPanduanTabPanel, {
    updateNavbarPage(session, "navbar", selected = "Panduan Penggunaan")
  })
  
  # Analysis Button event
  observeEvent(input$run, {
    # Reset error message and show loading message
    output$error_message <- renderText({
      NULL
    })
    output$loading_message <- renderText({
      "Loading..."
    })
    
    shinyjs::show("loading_message")
    
    # Validate inputs
    error_message <- NULL
    if (is.null(input$units) || input$units < 1 || input$units > 512) {
      error_message <- "Units harus bernilai antara 1 dan 512."
    } else if (is.null(input$epochs) || input$epochs < 10 || input$epochs > 200) {
      error_message <- "Epochs harus bernilai antara 10 hingga 200."
    }
    
    if (!is.null(error_message)) {
      output$error_message <- renderText({
        paste("Error: ", error_message)
      })
      shinyjs::hide("loading_message")
      return(NULL)
    }
    
    req(input$file)
    train_size <- input$train_size
    units <- input$units
    dropout <- input$dropout
    learning_rate <- input$learning_rate
    epochs <- input$epochs
    tune <- input$tune
    
    train_size <- as.double(train_size)
    units <- as.integer(units)
    dropout <- as.double(dropout)
    learning_rate <- as.double(learning_rate)
    epochs <- as.integer(epochs)
    
    tryCatch({
      result = list()
      if (tune) {
        result <- main(file_path = input$file$datapath, train_size = train_size, units = units, 
                       dropout = dropout, lr = learning_rate, use_tuning = TRUE, epochs = epochs)
        
      } else {
        result <- main(file_path = input$file$datapath, train_size = train_size, units = units, 
                       dropout = dropout, lr = learning_rate, use_tuning = FALSE, epochs = epochs)
      }
      history_df <- result[[1]]
      df_train_metrics <- result[[2]]
      df_eval_metrics <- result[[3]]
      df_actual <- result[[4]]
      df_predicted <- result[[5]]
      df_comparison <- result[[6]]
      units <- result[[7]]
      dropout <- result[[8]]
      lr <- result[[9]]
      
      # Display parameters used
      params_used <- paste(
        "Proporsi Data Training: ", train_size, 
        "\nJumlah Units dalam layes LSTM: ", units, 
        "\nPersentase Dropout: ", dropout, 
        "\nLearning Rate: ", lr, 
        "\nEpochs: ", epochs,
        "\nTuning Hyperparameter: ", ifelse(tune, "Ya", "Tidak")
      )
      
      output$params_used <- renderText({
        params_used
      })
      
      history_df$epoch <- seq_len(nrow(history_df))
      model_loss_plot <- ggplot(history_df, aes(x = epoch)) +
        geom_line(aes(y = loss, color = "Training Loss")) +
        labs(title = "Model Loss Over Epochs", x = "Epoch", y = "Loss") +
        scale_color_manual("", breaks = c("Training Loss"),
                           values = c("Training Loss" = "blue")) + 
        theme_minimal()
      
      output$model_loss <- renderPlot({
        model_loss_plot
      })
      
      output$download_loss <- downloadHandler(
        filename = function() { "training_loss.csv" },
        content = function(file) {
          write.csv(history_df, file)
        }
      )
      
      output$download_plot <- downloadHandler(
        filename = function() { "model_loss.png" },
        content = function(file) {
          ggsave(file, plot = model_loss_plot, device = "png", width = 8, height = 6)
        }
      )
      
      # Remove "Actual " and "Predicted " prefixes from column names
      df_actual1 = df_actual
      df_predicted1 = df_predicted
      colnames(df_actual1) <- gsub("^Actual ", "", colnames(df_actual1))
      colnames(df_predicted1) <- gsub("^Predicted ", "", colnames(df_predicted1))
      
      output$plots_ui <- renderUI({
        plot_output_list <- lapply(colnames(df_actual1), function(var) {
          plotname <- paste("plot", var, sep = "_")
          downloadname <- paste("actual_vs_predicted", var, sep = "_")
          div(
            plotlyOutput(plotname),
            downloadButton(downloadname, "Unduh Plot")
          )
        })
        do.call(tagList, plot_output_list)
      })
      
      observe({
        lapply(colnames(df_actual1), function(var) {
          output[[paste("plot", var, sep = "_")]] <- renderPlotly({
            p <- ggplot() +
              geom_line(data = data.frame(Date = as.Date(rownames(df_actual1)), Value = df_actual1[[var]]),
                        aes(x = Date, y =Value, colour="Aktual"), size = 1) +
              geom_line(data = data.frame(Date = as.Date(rownames(df_predicted1)), Value = df_predicted1[[var]]),
                        aes(x = Date, y = Value, colour="Prediksi"), size = 1) +
              scale_color_manual(name = "Nilai", values = c("Aktual" = "darkblue", "Prediksi" = "red")) +
              ggtitle(paste("Actual vs Predicted for", var)) +
              xlab("Date") + ylab(var) +
              theme_minimal()
            ggplotly(p)
          })
          
          output[[paste("actual_vs_predicted", var, sep = "_")]] <- downloadHandler(
            filename = function() { paste("actual_vs_predicted_", var, ".png", sep = "") },
            content = function(file) {
              p <- ggplot() +
                geom_line(data = data.frame(Date = as.Date(rownames(df_actual1)), Value = df_actual1[[var]]),
                          aes(x = Date, y = Value, colour="Aktual"), size = 1) +
                geom_line(data = data.frame(Date = as.Date(rownames(df_predicted1)), Value = df_predicted1[[var]]),
                          aes(x = Date, y = Value, colour="Prediksi"), size = 1) +
                scale_color_manual(name = "Nilai", values = c("Aktual" = "darkblue", "Prediksi" = "red")) +
                ggtitle(paste("Actual vs Predicted for", var)) +
                xlab("Date") + ylab(var) +
                theme_minimal()
              ggsave(file, plot = p, device = "png", width = 8, height = 6)
            }
          )
        })
      })
      
      output$train_evaluation_table <- renderDT({
        # datatable(read.csv("result_table/evaluation_metrics.csv"))
        datatable(df_train_metrics)
      })
      
      output$download_train_evaluation <- downloadHandler(
        filename = function() { "train_evaluation_metrics.csv" },
        content = function(file) {
          write.csv(df_train_metrics, file)
        }
      )
      
      output$test_evaluation_table <- renderDT({
        # datatable(read.csv("result_table/evaluation_metrics.csv"))
        datatable(df_eval_metrics)
      })
      
      output$download_test_evaluation <- downloadHandler(
        filename = function() { "test_evaluation_metrics.csv" },
        content = function(file) {
          write.csv(df_eval_metrics, file)
        }
      )
      
      output$comparison_table <- renderDT({
        # datatable(read.csv("result_table/comparison_results.csv"))
        datatable(df_comparison)
      })
      
      output$download_comparison <- downloadHandler(
        filename = function() { "comparison_results.csv" },
        content = function(file) {
          write.csv(df_comparison, file)
          # file.copy("result_table/comparison_results.csv", file)7]
        }
      )
      
      output$error_message <- renderText({
        NULL
      })
      
      # Hide loading message
      shinyjs::hide("loading_message")
      
      # Unlock the "Hasil" tab and switch to it after successful analysis
      shinyjs::enable(selector = 'a[data-value="Hasil"]')
      updateNavbarPage(session, "navbar", selected = "Hasil")
      
    }, error = function(e) {
      output$error_message <- renderText({
        paste("Error: ", e$message)
      })
      shinyjs::hide("loading_message")
    })
  })
}

shinyApp(ui, server)

