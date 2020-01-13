package demo.spark;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import static org.apache.spark.sql.functions.*;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.util.DoubleAccumulator;



public class Main {
  public static void main(String[] args) {
  	System.out.println("Started...");
  	System.setProperty("hadoop.home.dir", "C:\\data\\hadoop");
  	
  	SparkSession spark = SparkSession
    	      .builder()
    	      .master("local")
    	      .appName("Spark_Demo_KMeans")
    	      .getOrCreate();
  	spark.sparkContext().setLogLevel("ERROR");
  	
  	JavaSparkContext spContext = new JavaSparkContext(spark.sparkContext());
  	
  	Dataset<Row> autoDf = spark.read().option("header", "true")
								.csv("C:\\data\\mllib\\auto-data.csv");
				
  	autoDf.show(5);
  	autoDf.printSchema();
  	
  		
	//Create the schema for the data to be loaded into Dataset.
	StructType autoSchema = DataTypes
			.createStructType(new StructField[] {
					DataTypes.createStructField("DOORS", DataTypes.DoubleType, false),
					DataTypes.createStructField("BODY", DataTypes.DoubleType, false),
					DataTypes.createStructField("HP", DataTypes.DoubleType, false),
					DataTypes.createStructField("RPM", DataTypes.DoubleType, false),
					DataTypes.createStructField("MPG", DataTypes.DoubleType, false) 
				});
	
	JavaRDD<Row> rdd1 = autoDf.toJavaRDD().repartition(2);
	
	//Function to map.
	JavaRDD<Row> rdd2 = rdd1.map( new Function<Row, Row>() {

		public Row call(Row iRow) throws Exception {
			
			double doors = ( iRow.getString(3).equals("two") ? 1.0 : 2.0);
			double body = ( iRow.getString(4).equals("sedan") ? 1.0 : 2.0);
			
			Row retRow = RowFactory.create( doors, body,
							Double.valueOf(iRow.getString(7)),
							Double.valueOf(iRow.getString(8)), 
							Double.valueOf(iRow.getString(9)) );
			
			return retRow;
		}

	});
	
	
	Dataset<Row> autoCleansedDf = spark.createDataFrame(rdd2, autoSchema);
	System.out.println("Transformed Data :");
	autoCleansedDf.show(5);
	
	//Convert features into dense vector
	final DoubleAccumulator rowId = spContext.sc().doubleAccumulator();
	rowId.setValue(1);

	//Perform center-and-scale and create a vector
	JavaRDD<Row> rdd3 = autoCleansedDf.toJavaRDD().repartition(2);
	JavaRDD<LabeledPoint> rdd4 = rdd3.map( new Function<Row, LabeledPoint>() {

		public LabeledPoint call(Row iRow) throws Exception {
							
			double id= rowId.value();
			rowId.setValue(rowId.value()+1);
			
			LabeledPoint lp = new LabeledPoint( id,
					Vectors.dense(iRow.getDouble(0), iRow.getDouble(1), iRow.getDouble(2), iRow.getDouble(3), iRow.getDouble(4)));
			
			return lp;
		}

	});

	Dataset<Row> autoVector = spark.createDataFrame(rdd4, LabeledPoint.class );
	System.out.println("Test vector :" + autoVector.count());
	autoVector.show(5);
	
	//Use standard scaling
	StandardScaler scaler = new StandardScaler()
		      .setInputCol("features")
		      .setOutputCol("scaledFeatures")
		      .setWithStd(true)
		      .setWithMean(true);

    // Compute summary statistics by fitting the StandardScaler
    StandardScalerModel scalerModel = scaler.fit(autoVector);

    // Normalize each feature to have unit standard deviation.
    Dataset<Row> scaledData = scalerModel.transform(autoVector);
    System.out.println("Scaled Data :");
    scaledData.show(5);
    
    //Perform KMeans Clustering with K = 4
	
	KMeans kmeans = new KMeans()
						.setK(4)
						.setSeed(1L)
						.setFeaturesCol("scaledFeatures");
	
	KMeansModel model = kmeans.fit(scaledData);
	Dataset<Row> predictions = model.transform(scaledData);
	
	// Evaluate clustering by computing Within Set Sum of Squared Errors
	double WSSSE = model.computeCost(scaledData);
	System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
	
	System.out.println("Groupings : ");
	predictions.show(5);
	
	System.out.println("Groupings Summary : ");
	predictions.groupBy(col("prediction")).count().show();
    

     System.out.println("Finished...");
      
  }
}
