import sys
import pandas as pd
import random
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QFileDialog, QTableView, QMessageBox, 
                               QTabWidget, QTextEdit, QSplitter, QStatusBar, QProgressBar,
                               QComboBox, QGridLayout, QGroupBox, QSizePolicy)
from PySide6.QtGui import QImage, QPixmap, QStandardItemModel, QStandardItem, QAction
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import QHeaderView

from model import (load_model, train_model, predict, save_model, 
                  get_model_metrics, generate_analysis_plot, 
                  is_trained, required_columns)

ML_FACTS = [
    "Модели машинного обучения могут выявлять закономерности, неочевидные для человека.",
    "Первая нейронная сеть была создана в 1958 году Фрэнком Розенблаттом.",
    "Глубокое обучение - это подраздел машинного обучения, использующий многослойные нейронные сети.",
    "Переобучение возникает, когда модель слишком хорошо запоминает обучающие данные и плохо работает на новых.",
    "Термин 'большие данные' относится к наборам данных, слишком большим для традиционной обработки.",
    "Обучение с подкреплением используется для тренировки агентов, принимающих последовательные решения.",
    "Компромисс между смещением и дисперсией - ключевая концепция в машинном обучении.",
    "Обработка естественного языка (NLP) позволяет машинам понимать человеческую речь.",
    "Метод опорных векторов (SVM) эффективен в пространствах высокой размерности.",
    "Для многих задач качество данных важнее выбора алгоритма.",
    "Алгоритм k-ближайших соседей (KNN) - один из простейших алгоритмов машинного обучения.",
    "Рекуррентные нейронные сети (RNN) особенно эффективны для обработки последовательных данных.",
    "Сверточные нейронные сети (CNN) революционизировали компьютерное зрение.",
    "Генеративно-состязательные сети (GAN) могут создавать реалистичные изображения людей, которых не существует.",
    "Трансформеры стали основой для современных языковых моделей, таких как GPT.",
    "Машинное обучение используется в медицинской диагностике для выявления заболеваний на ранних стадиях.",
    "Рекомендательные системы Netflix и Spotify используют алгоритмы машинного обучения.",
    "Автономные автомобили полагаются на компьютерное зрение и глубокое обучение.",
    "Обработка естественного языка позволяет чат-ботам понимать и генерировать человеческую речь.",
    "Машинное обучение помогает предсказывать колебания фондового рынка.",
    "Кластеризация используется для сегментации клиентов без предварительных меток.",
    "Ансамбли моделей (например, случайный лес) часто превосходят одиночные модели.",
    "Градиентный бустинг (XGBoost, LightGBM) - мощный метод для табличных данных.",
    "Перенос обучения позволяет использовать предобученные модели для новых задач.",
    "Регуляризация помогает предотвратить переобучение моделей.",
    "Машинное обучение используется для оптимизации логистических маршрутов.",
    "Системы распознавания лиц основаны на глубоком обучении.",
    "Генетические алгоритмы вдохновлены принципами естественного отбора.",
    "Машинное обучение помогает ученым анализировать данные с Большого адронного коллайдера.",
    "AI-модели могут сочинять музыку и создавать произведения искусства.",
    "Машинное обучение используется для прогнозирования погоды и климатических изменений.",
    "Автокодировщики (autoencoders) применяются для уменьшения размерности данных.",
    "Q-обучение - популярный алгоритм в обучении с подкреплением.",
    "Машинное обучение помогает обнаруживать мошеннические транзакции в банковской сфере.",
    "Анализ тональности текста позволяет определить эмоциональную окраску сообщений.",
    "Машинное обучение используется в сельском хозяйстве для оптимизации урожая.",
    "Глубокое обучение помогло решить проблему сворачивания белков, над которой ученые бились 50 лет.",
    "AI-модели могут генерировать реалистичные человеческие голоса.",
    "Машинное обучение используется для персонализации рекламы в интернете.",
    "Нейронные сети могут быть визуализированы с помощью инструментов типа TensorBoard.",
]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bank Marketing Prediction")
        self.resize(1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
      
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
      
        self.data_tab = QWidget()
        self.data_tab_layout = QVBoxLayout(self.data_tab)
        
        control_layout = QHBoxLayout()
        
        self.btn_load_model = QPushButton("Load Model")
        self.btn_load_model.clicked.connect(self.load_model)
        control_layout.addWidget(self.btn_load_model)
        
        self.btn_train_model = QPushButton("Train Model")
        self.btn_train_model.clicked.connect(self.train_model)
        control_layout.addWidget(self.btn_train_model)
        
        self.btn_load_data = QPushButton("Load Data")
        self.btn_load_data.clicked.connect(self.load_data)
        control_layout.addWidget(self.btn_load_data)
        
        self.btn_predict = QPushButton("Predict")
        self.btn_predict.clicked.connect(self.predict)
        control_layout.addWidget(self.btn_predict)
        
        self.btn_save_results = QPushButton("Save Results")
        self.btn_save_results.clicked.connect(self.save_results)
        control_layout.addWidget(self.btn_save_results)
        
        self.data_tab_layout.addLayout(control_layout)
    
        self.table_view = QTableView()
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_tab_layout.addWidget(self.table_view)
      
        self.analysis_tab = QWidget()
        self.analysis_tab_layout = QHBoxLayout(self.analysis_tab)
        
        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(5, 5, 5, 5)
        
        self.variable_selection_group = QGroupBox("Variable Selection")
        self.variable_selection_layout = QVBoxLayout()
        
        self.variable_label = QLabel("Select Variable:")
        self.variable_combobox = QComboBox()
        self.variable_combobox.setMinimumWidth(200)
        
        self.variable_selection_layout.addWidget(self.variable_label)
        self.variable_selection_layout.addWidget(self.variable_combobox)
        
        self.btn_generate_plot = QPushButton("Generate Analysis Plot")
        self.btn_generate_plot.clicked.connect(self.generate_plot)
        self.variable_selection_layout.addWidget(self.btn_generate_plot)
        
        self.variable_selection_group.setLayout(self.variable_selection_layout)
        left_panel.addWidget(self.variable_selection_group)
        
        self.facts_group = QGroupBox("Machine Learning Facts")
        self.facts_layout = QVBoxLayout()
        self.fact_text = QTextEdit()
        self.fact_text.setReadOnly(True)
        self.fact_text.setMinimumHeight(200)
        self.fact_text.setStyleSheet("font-size: 14px;")
        self.facts_layout.addWidget(self.fact_text)
        
        self.btn_new_fact = QPushButton("New Fact")
        self.btn_new_fact.clicked.connect(self.show_random_fact)
        self.facts_layout.addWidget(self.btn_new_fact)
        
        self.facts_group.setLayout(self.facts_layout)
        left_panel.addWidget(self.facts_group)
        
        left_panel.addStretch()
        
        right_panel = QVBoxLayout()
        self.plot_group = QGroupBox("Analysis Plot")
        self.plot_layout = QVBoxLayout()
        self.plot_label = QLabel()
        self.plot_label.setAlignment(Qt.AlignCenter)
        self.plot_label.setMinimumSize(600, 500)
        self.plot_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_layout.addWidget(self.plot_label)
        self.plot_group.setLayout(self.plot_layout)
        right_panel.addWidget(self.plot_group)
 
        self.analysis_tab_layout.addLayout(left_panel, 1)
        self.analysis_tab_layout.addLayout(right_panel, 3)
        
        self.tabs.addTab(self.data_tab, "Data")
        self.tabs.addTab(self.analysis_tab, "Model Analysis")
   
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.current_data = None
        self.results = None
        self.available_columns = []

        self.show_random_fact()

        self.load_stylesheet()

    def load_stylesheet(self):
        try:
            with open("./style.qss", "r") as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            print(f"Не удалось загрузить стиль: {e}")

    def show_random_fact(self):
        fact = random.choice(ML_FACTS)
        self.fact_text.setText(f"<p style='font-size: 16px;'><b>Did you know?</b></p><p>{fact}</p>")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "Joblib Files (*.joblib)"
        )
        if file_path:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0) 
            
            success, message = load_model(file_path)
            
            self.progress_bar.setVisible(False)
            
            if success:
                self.update_status_bar()
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.critical(self, "Error", message)

    def train_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Training Data", "", "CSV Files (*.csv)"
        )
        if file_path:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            
            success, message = train_model(file_path)
            
            self.progress_bar.setVisible(False)
            
            if success:
                save_success, save_msg = save_model()
                if save_success:
                    self.update_status_bar()
                    QMessageBox.information(self, "Success", f"{message}\n{save_msg}")
                else:
                    QMessageBox.warning(self, "Warning", f"{message}\nBut failed to save: {save_msg}")
            else:
                QMessageBox.critical(self, "Error", message)

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Data", "", "CSV Files (*.csv)"
        )
        if file_path:
            try:
                self.current_data = pd.read_csv(file_path, sep=";")
                self.display_data(self.current_data)
                self.status_bar.showMessage(f"Data loaded: {file_path}")
                
                self.available_columns = [col for col in self.current_data.columns 
                                         if col != 'predicted_probability']
                self.update_variable_combobox()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def predict(self):
        if self.current_data is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        success, message, results = predict(self.current_data)
        
        self.progress_bar.setVisible(False)
        
        if success:
            self.results = results
            self.display_data(results)
            self.status_bar.showMessage(message)
            
            self.available_columns = [col for col in self.results.columns 
                                     if col != 'predicted_probability']
            self.update_variable_combobox()
            
            self.tabs.setCurrentIndex(0)
        else:
            QMessageBox.critical(self, "Error", message)

    def save_results(self):
        if self.results is None:
            QMessageBox.warning(self, "Warning", "No results to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "CSV Files (*.csv)"
        )
        if file_path:
            try:
                self.results.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", "Results saved successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")

    def display_data(self, data):
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(data.columns.tolist())
        
        sample = data.head(1000)
        
        for _, row in sample.iterrows():
            items = []
            for value in row:
                if isinstance(value, float):
                    item = QStandardItem(f"{value:.4f}")
                else:
                    item = QStandardItem(str(value))
                items.append(item)
            model.appendRow(items)
        
        self.table_view.setModel(model)

    def update_variable_combobox(self):
        """Обновление выпадающего списка переменных"""
        self.variable_combobox.clear()
        self.variable_combobox.addItem("")
        self.variable_combobox.addItems(self.available_columns)

    def generate_plot(self):
        """Генерация графика для выбранной переменной"""
        if self.results is None:
            QMessageBox.warning(self, "Warning", "No data available. Please load data and predict first.")
            return
            
        variable = self.variable_combobox.currentText()
        if not variable or variable == "":
            QMessageBox.warning(self, "Warning", "Please select a variable.")
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.plot_label.clear()
        
        plot = generate_analysis_plot(self.results, variable)
        
        self.progress_bar.setVisible(False)
        
        if plot:
            img_data = plot.getvalue()
            pixmap = QPixmap()
            pixmap.loadFromData(img_data)
            self.plot_label.setPixmap(pixmap.scaled(
                self.plot_label.width(), 
                self.plot_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.show_random_fact()
        else:
            QMessageBox.warning(self, "Warning", f"Failed to generate plot for {variable}")

    def update_status_bar(self):
        if not is_trained():
            return
            
        success, message, metrics = get_model_metrics()
        if not success:
            return
            
        accuracy = metrics.get('accuracy', 0)
        roc_auc = metrics.get('roc_auc', 0)
        self.status_bar.showMessage(
            f"Model metrics: Accuracy={accuracy:.4f}, ROC-AUC={roc_auc:.4f}"
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())