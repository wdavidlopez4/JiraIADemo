using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace JiraIADemo
{
    class Program
    {
        private static string _appPath => Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
        private static string _trainDataPath => Path.Combine(_appPath, "Data", "issues_train.txt");
        private static string _testDataPath => Path.Combine(_appPath, "Data", "issues_test.txt");
        private static string _modelPath => Path.Combine(_appPath, "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            Console.WriteLine(_trainDataPath);

            //configurar
            _mlContext = new MLContext(seed: 0);
            _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);
            var pipeline = ProcessData();
            IEstimator<ITransformer> trainingPipeline;

            string titulo, descripcion;

            if(File.Exists(_modelPath) == false)
            {
                //entrenar si no existe modelo
                trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);

                //evaluar
                Evaluate(_trainingDataView.Schema);

                //guardar el modelo entrenado y evaluado
                SaveModelAsFile(_mlContext, _trainingDataView.Schema, _trainedModel);
            }

            //predicion del caso que nos pasen
            Console.WriteLine("Ingrese un titulo: ");
            titulo = Console.ReadLine();

            Console.WriteLine("Ingrese una descripcion: ");
            descripcion = Console.ReadLine();

            PredictIssue(titulo, descripcion);
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "SolucionAplicada", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext); //para pequeños conjuntos de datos

            return pipeline;
        }

        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = trainingPipeline.Fit(trainingDataView);

            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);

            //probarlo
            var issue = new GitHubIssue()
            {
                Title = "EN LAS NOTAS DEBITO Y CREDITO DE CXC, SE REPITE EL ANCABEZADO POR CADA TRASACCION",
                Description = "ENTRADA: PASO 1. USUARIO DEL SISTEMA: 18420303 SEDE: HOSPITAL PRINCIPAL " +
                    "//CLIC EN ACEPTAR PROCESO: PASO 2.  IR AL PILAR AMARILLO ADMINISTRATIVO PASO 3. IR AL " +
                    "MODULO CARTERA PASO 4. ABRIR EL NODO NOTA CREDITO O NOTA DEBITO ACTIVAR FECHA INICIAL " +
                    "01-02-2020  FECHA FINAL: FECHA ACTUAL CLIC EN EL BOTON BUSCAR SELECCIONAR LA NOTA DEBITO  " +
                    "1083  CLIC EN VER Y CLIC EN IMPRIMIR EN NOTA CREDITO HACER EL MISMO EJERCICIO PERO ESCOGER " +
                    "LA NOTA CREDITO 79752 RESULTADO OBTENIDO: SE MUESTRA EL ENCABEZADO POR NUMERO DE CXC Y " +
                    "CONCEPTO POR SEPARADO REPITIENDO EL ENCABEZADO POR CADA TRANSACCION HACIENDO QUE SE IMPRIMA " +
                    "INFORMACION REPETIDA POR CADA TRANSACCION."
            };

            var prediction = _predEngine.Predict(issue);

            Console.WriteLine($"=============== Predicción única modelo recién entrenado - Result: {prediction.SolucionAplicada} ===============");
            Console.WriteLine($"--------------- titulo : {issue.Title} -------------------------------------------------------------");
            Console.WriteLine($"--------------- descripcion : {issue.Description} -------------------------------------------------------------");

            return trainingPipeline;
        }

        public static void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);

            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));

            //simplemente mostramos las metricas
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Métricas para el modelo de clasificación de clases múltiples: datos de prueba     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Microprecisión:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       Macroprecisión:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");
        }

        private static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
        }

        /// <summary>
        /// hacemos la predicion
        /// </summary>
        private static void PredictIssue(string titulo, string descripcion)
        {
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
            GitHubIssue singleIssue = new GitHubIssue() 
            {
                Title = titulo,
                Description = descripcion
            };

            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);

            var prediction = _predEngine.Predict(singleIssue);

            Console.WriteLine($"=============== Predicción única - Resultado: {prediction.SolucionAplicada} ===============");
            Console.WriteLine($"titulo - Resultado: {singleIssue.Title} ===============");
            Console.WriteLine($"descripcion - Resultado: {singleIssue.Description} ===============");
        }

        public class GitHubIssue
        {
            [LoadColumn(0)]
            public string ID { get; set; }
            [LoadColumn(1)]
            public string SolucionAplicada { get; set; }
            [LoadColumn(2)]
            public string Title { get; set; }
            [LoadColumn(3)]
            public string Description { get; set; }
        }

        public class IssuePrediction
        {
            [ColumnName("PredictedLabel")]
            public string SolucionAplicada;
        }
    }
}
