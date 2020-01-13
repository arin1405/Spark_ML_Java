package com.demo;

import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.JavaSparkContext;

public class Main {

	public static void main(String[] args) {
		
		//System.setProperty("hadoop.home.dir", "C:\\data\\hadoop");
		
		SparkSession spSession = SparkSession
				.builder()
				//.master("local")
				.master("yarn")
				.appName("Demo_Reg")
				.getOrCreate();
		
		spSession.sparkContext().setLogLevel("ERROR");
		
		Dataset<Row> autoDf = spSession.read()
				.option("header", "true")
				.csv("/tmp/datasets/auto-miles-per-gallon.csv");
		autoDf.show(5);
		autoDf.printSchema();
		
		//Create the schema for the data to be loaded into Dataset.
		StructType autoSchema = DataTypes
				.createStructType(new StructField[] {
						DataTypes.createStructField("MPG", DataTypes.DoubleType, false),
						DataTypes.createStructField("CYLINDERS", DataTypes.DoubleType, false),
						DataTypes.createStructField("DISPLACEMENT", DataTypes.DoubleType, false),
						DataTypes.createStructField("HP", DataTypes.DoubleType, false),
						DataTypes.createStructField("WEIGHT", DataTypes.DoubleType, false),
						DataTypes.createStructField("ACCELERATOIN", DataTypes.DoubleType, false),
						DataTypes.createStructField("MODELYEAR", DataTypes.DoubleType, false),
						DataTypes.createStructField("NAME", DataTypes.StringType, false),
				});
		
		JavaSparkContext spContext = 
				JavaSparkContext.fromSparkContext(spSession.sparkContext());
		
		//Missing Value treatment: Broadcast the default value for HP
		final Broadcast<Double> avgHP = spContext.broadcast(80.0);
		
		JavaRDD<Row> rdd1 = autoDf.toJavaRDD().repartition(2);
		
		JavaRDD<Row> rdd2 = rdd1.map(new Function<Row, Row>(){
			public Row call(Row iRow) throws Exception{
				
				double hp = (iRow.getString(3).equals("?") ? 
						avgHP.value() : Double.valueOf(iRow.getString(3)));
				
				Row retRow = RowFactory.create(Double.valueOf(iRow.getString(0)),
						Double.valueOf(iRow.getString(1)),
						Double.valueOf(iRow.getString(2)),
						Double.valueOf(hp),
						Double.valueOf(iRow.getString(4)),
						Double.valueOf(iRow.getString(5)),
						Double.valueOf(iRow.getString(6)),
						iRow.getString(7)
						);
				return retRow;
			}
		});
				
		Dataset<Row> autoCleansedDf = spSession.createDataFrame(rdd2, autoSchema);
		System.out.println("Transformed data...");
		autoCleansedDf.show(5);
		
		//Correlation Analysis: Calculate correlation between MPG (target variable) and other variables
		for (StructField field : autoSchema.fields()) {
			if(!field.dataType().equals(DataTypes.StringType)) {
				System.out.println("Correlation between MPG and "+field.name()
				+ " = " + autoCleansedDf.stat().corr("MPG", field.name()));
			}
		}
		
		/*
			Convert data to labeled Point structure for preparing dataset for ML analysis. 
		    Label is the target variable (MPG).
			Input columns (independent variables) are converted into a dense vector.
		*/
		
		JavaRDD<Row> rdd3 = autoCleansedDf.toJavaRDD().repartition(2);
		
		JavaRDD<LabeledPoint> rdd4 = rdd3.map(new Function<Row, LabeledPoint>(){
			public LabeledPoint call(Row iRow) throws Exception{
				
				LabeledPoint lp = new LabeledPoint(iRow.getDouble(0), 
						Vectors.dense(iRow.getDouble(2),
								iRow.getDouble(4),
								iRow.getDouble(5)));
				return lp;
			}
		});
		
		Dataset<Row> autoLp = spSession.createDataFrame(rdd4,  LabeledPoint.class);
		
		System.out.println("Labeled data...");
		autoLp.show(5);
		
		//Split data into training and test sets
		
		Dataset<Row>[] splits = autoLp.randomSplit(new double[] {0.8, 0.2});
		Dataset<Row> trainingData = splits[0];
		Dataset<Row> testData = splits[1];
		
		//Use standard scaling to scale the data. Fit training data but transform both training and test data for scaling.
		StandardScaler scaler = new StandardScaler()
				.setInputCol("features")
				.setOutputCol("scaledFeatures")
				.setWithMean(true)
				.setWithStd(true);
		
		StandardScalerModel scalerModel = scaler.fit(trainingData);
		
		Dataset<Row> scaledTrainingData = scalerModel.transform(trainingData);
		Dataset<Row> scaledTestData = scalerModel.transform(testData);
		
		System.out.println("Scaled training data..");
		scaledTrainingData.show(5);
		
		//Linear regression model. You may play with 'setRegParam' by putting different regularization values.
		
		LinearRegression lr = new LinearRegression()
				.setFeaturesCol("scaledFeatures")
				.setRegParam(1);
		
		LinearRegressionModel lrModel = lr.fit(scaledTrainingData);
		
		System.out.println("Coefficients: "
				+lrModel.coefficients() + " Intercept: " + lrModel.intercept());
		
		Dataset<Row> predictions = lrModel.transform(scaledTestData);
		
		predictions.select("label", "prediction", "features").show(5);
		
		//Evaluate regression result using R2. You may also try with rmse.
		
		RegressionEvaluator evaluator = new RegressionEvaluator()
				.setLabelCol("label")
				.setPredictionCol("prediction")
				.setMetricName("r2"); //rmse
		
		double r2 = evaluator.evaluate(predictions);
		System.out.println("R2 on test data: " + r2);
		spSession.close();
		System.out.println("Finish");
				
		
		
	}

}
