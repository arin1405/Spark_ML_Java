package demo.spark;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import static org.apache.spark.sql.functions.*;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		//System.setProperty("hadoop.home.dir", "C:\\data\\hadoop");
		
		SparkSession spark = SparkSession
				.builder()
				//.master("local")
				.master("yarn")
				.appName("Demo_classification")
				.getOrCreate();
		
		spark.sparkContext().setLogLevel("ERROR");
		
		StructType twSchema = DataTypes
				.createStructType(new StructField[] {
						DataTypes.createStructField("label", DataTypes.DoubleType, false),
						DataTypes.createStructField("tweet", DataTypes.StringType, false),
				});
		
		Dataset twDf = spark.read().option("header", "true")
				.csv("/tmp/datasets/tweets.csv");
		twDf.show(5);
		twDf.printSchema();
		
		//Data Cleaning
		
		JavaRDD<Row> rdd1 = twDf.toJavaRDD().repartition(2);
		
		JavaRDD<Row> rdd2 = rdd1.map(new Function<Row, Row>(){
			public Row call(Row iRow) throws Exception{
				double sentiment = (iRow.getString(1).contains("1") ? 1.0 : 0.0);
				
				Row retRow = RowFactory.create(sentiment, iRow.getString(2));
				return retRow;
			}
		});
		
		Dataset<Row> twCleanDf = spark.createDataFrame(rdd2, twSchema);
		System.out.println("Clean Data");
		System.out.println(twCleanDf);
		
		//Split data into training and test sets
		Dataset<Row>[] splits = twCleanDf.randomSplit(new double[] {0.7, 0.3});
		Dataset<Row> trainingData = splits[0];
		Dataset<Row> testData = splits[1];
		
		//Pipeline
		Tokenizer tokenizer = new Tokenizer()
				.setInputCol("tweet")
				.setOutputCol("words");
		
		HashingTF hashingTF = new HashingTF()
				.setInputCol("words")
				.setOutputCol("rawFeatures");
		
		IDF idf = new IDF()
				.setInputCol("rawFeatures")
				.setOutputCol("features");
		
		RandomForestClassifier rf = new RandomForestClassifier()
				.setLabelCol("label")
				.setFeaturesCol("features")
				.setNumTrees(50)
				.setMaxDepth(10);
		
		Pipeline pipeline = new Pipeline()
				.setStages(new PipelineStage[]
						{tokenizer, hashingTF, idf, rf});
		
		ParamMap[] paramGrid = new ParamGridBuilder()
				.addGrid(hashingTF.numFeatures(), new int[] {10, 100, 1000})
				.addGrid(rf.numTrees(), new int[] {30, 50})
				.addGrid(rf.maxDepth(), new int[] {5, 10})
				.build();
		
		CrossValidator cv = new CrossValidator()
				.setEstimator(pipeline)
				.setEvaluator(new BinaryClassificationEvaluator())
				.setEstimatorParamMaps(paramGrid)
				.setNumFolds(2)
				.setParallelism(2);
		
		CrossValidatorModel cvModel = cv.fit(trainingData);
		Dataset<Row> predictions = cvModel.transform(testData);
		
		System.out.println("Best Model");
		System.out.println(cvModel.bestModel());
		
		System.out.println("Result Sample");
		predictions.show(5);
		
		System.out.println("Confusion Matrix: ");
		predictions.groupBy(col("label"), col("prediction")).count().show();
		
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("label")
				.setPredictionCol("prediction")
				.setMetricName("accuracy");
		
		double accuracy = evaluator.evaluate(predictions);
		System.out.println("Accuracy: "+Math.round(accuracy*100)+" %");
		System.out.println("Finish");
	}

}
