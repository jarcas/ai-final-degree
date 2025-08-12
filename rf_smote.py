# Detección de Ataques con SMOTE + Random Forest
# Dataset: OCPPFlowMeter con clases desbalanceadas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           f1_score, balanced_accuracy_score)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
sns.set_palette("husl")

class AttackDetectionML:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.smote = SMOTETomek(random_state=13)
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_explore_data(self, filepath):
        """Carga y explora el dataset"""
        print("=== CARGANDO DATASET ===")
        self.df = pd.read_csv(filepath)
        
        print(f"Dimensiones del dataset: {self.df.shape}")
        print(f"\nTipos de datos:")
        print(self.df.dtypes)
        
        print(f"\nValores nulos por columna:")
        print(self.df.isnull().sum())
        
        print(f"\nDistribución de clases:")
        class_dist = self.df.iloc[:, -1].value_counts()
        print(class_dist)
        
        # Visualizar distribución de clases
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        class_dist.plot(kind='bar', color=['#FF6B6B', '#4ECDC4'])
        plt.title('Distribución de Clases')
        plt.xlabel('Clase')
        plt.ylabel('Cantidad')
        plt.xticks(rotation=0)
        
        plt.subplot(1, 2, 2)
        plt.pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%',
                colors=['#FF6B6B', '#4ECDC4'])
        plt.title('Proporción de Clases')
        
        plt.tight_layout()
        plt.show()
        
        return self.df
    
    def preprocess_data(self):
        """Preprocesa los datos"""
        print("\n=== PREPROCESAMIENTO ===")
        
        columns_to_drop = ['flow_id', 'flow_start_timestamp', 'flow_end_timestamp',
                   'src_ip', 'dst_ip', 'src_port', 'dst_port']
        self.df = self.df.drop(columns=columns_to_drop)  # Features

        
        # Separar características y target
        X = self.df.iloc[:, :-1]  # Todas las columnas excepto la última
        y = self.df.iloc[:, -1]   # Última columna (target)
        
        print(f"Características: {X.shape[1]} columnas")
        print(f"Target: {y.name if hasattr(y, 'name') else 'target'}")
        
        # Codificar variable target si es categórica
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
            print(f"Clases codificadas: {dict(zip(self.label_encoder.classes_, 
                                               self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        # Manejar valores nulos
        if X.isnull().sum().sum() > 0:
            print("Rellenando valores nulos con la mediana...")
            X = X.fillna(X.median())
        
        # Dividir en train/test estratificado
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Datos de entrenamiento: {self.X_train.shape}")
        print(f"Datos de prueba: {self.X_test.shape}")
        
        # Escalado de características
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Datos escalados correctamente")
        
    def apply_smote(self):
        """Aplica SMOTE para balancear las clases"""
        print("\n=== APLICANDO SMOTE ===")
        
        # Distribución antes de SMOTE
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"Distribución ANTES de SMOTE: {dict(zip(unique, counts))}")
        
        # Aplicar SMOTE + Tomek
        self.X_train_balanced, self.y_train_balanced = self.smote.fit_resample(
            self.X_train_scaled, self.y_train
        )
        
        # Distribución después de SMOTE
        unique, counts = np.unique(self.y_train_balanced, return_counts=True)
        print(f"Distribución DESPUÉS de SMOTE: {dict(zip(unique, counts))}")
        
        # Visualizar el efecto de SMOTE
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Antes de SMOTE
        unique_before, counts_before = np.unique(self.y_train, return_counts=True)
        axes[0].bar(unique_before, counts_before, color=['#FF6B6B', '#4ECDC4'])
        axes[0].set_title('Distribución ANTES de SMOTE')
        axes[0].set_xlabel('Clase')
        axes[0].set_ylabel('Cantidad')
        
        # Después de SMOTE
        unique_after, counts_after = np.unique(self.y_train_balanced, return_counts=True)
        axes[1].bar(unique_after, counts_after, color=['#FF6B6B', '#4ECDC4'])
        axes[1].set_title('Distribución DESPUÉS de SMOTE')
        axes[1].set_xlabel('Clase')
        axes[1].set_ylabel('Cantidad')
        
        plt.tight_layout()
        plt.show()
        
    def train_model(self):
        """Entrena el modelo Random Forest"""
        print("\n=== ENTRENANDO MODELO ===")
        
        # Entrenar con datos balanceados
        self.rf_model.fit(self.X_train_balanced, self.y_train_balanced)
        print("Modelo Random Forest entrenado exitosamente")
        
        # Validación cruzada estratificada
        cv_scores = cross_val_score(
            self.rf_model, self.X_train_balanced, self.y_train_balanced, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1_macro'
        )
        
        print(f"CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
    def evaluate_model(self):
        """Evalúa el modelo con métricas apropiadas para datos desbalanceados"""
        print("\n=== EVALUACIÓN DEL MODELO ===")
        
        # Predicciones
        y_pred = self.rf_model.predict(self.X_test_scaled)
        y_pred_proba = self.rf_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Métricas principales
        print("MÉTRICAS DE CLASIFICACIÓN:")
        print(classification_report(self.y_test, y_pred))
        
        print(f"Balanced Accuracy: {balanced_accuracy_score(self.y_test, y_pred):.4f}")
        print(f"AUC-ROC: {roc_auc_score(self.y_test, y_pred_proba):.4f}")
        
        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: Matriz de confusión
        plt.subplot(1, 3, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Attack'], 
                    yticklabels=['Normal', 'Attack'])
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        
        # Subplot 2: Curva ROC
        plt.subplot(1, 3, 2)
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc_score(self.y_test, y_pred_proba):.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        
        # Subplot 3: Curva Precision-Recall
        plt.subplot(1, 3, 3)
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Curva Precision-Recall')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        plt.tight_layout()
        plt.show()
        
    def feature_importance(self):
        """Analiza la importancia de las características"""
        print("\n=== IMPORTANCIA DE CARACTERÍSTICAS ===")
        
        # Obtener importancias
        feature_names = [f'Feature_{i}' for i in range(self.X_train.shape[1])]
        importances = self.rf_model.feature_importances_
        
        # Crear DataFrame para visualización
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Mostrar top 10
        print("TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:")
        print(feature_imp_df.head(10))
        
        # Visualizar
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_imp_df.head(15), x='importance', y='feature')
        plt.title('Top 15 Características más Importantes')
        plt.xlabel('Importancia')
        plt.tight_layout()
        plt.show()
        
        return feature_imp_df
    
    def run_complete_analysis(self, filepath):
        """Ejecuta el análisis completo"""
        try:
            # 1. Cargar y explorar datos
            self.load_and_explore_data(filepath)
            
            # 2. Preprocesar
            self.preprocess_data()
            
            # 3. Aplicar SMOTE
            self.apply_smote()
            
            # 4. Entrenar modelo
            self.train_model()
            
            # 5. Evaluar modelo
            self.evaluate_model()
            
            # 6. Importancia de características
            feature_importance = self.feature_importance()
            
            print("\n=== ANÁLISIS COMPLETADO ===")
            print("El modelo está listo para hacer predicciones")
            
            return self.rf_model, feature_importance
            
        except Exception as e:
            print(f"Error durante el análisis: {str(e)}")
            return None, None

# EJEMPLO DE USO:
if __name__ == "__main__":
    # Inicializar el analizador
    detector = AttackDetectionML()
    
    # Ejecutar análisis completo
    # NOTA: Reemplaza 'tu_archivo.csv' con la ruta de tu archivo
    modelo, importancias = detector.run_complete_analysis('20240625_Flooding_Heartbeat_filtered_ordered_OcppFlows_120_labelled.csv')
    
    # Hacer predicciones en nuevos datos (ejemplo)
    # nuevas_predicciones = modelo.predict(nuevos_datos_escalados)