package edu.uci.ics.classifiction;

import java.io.IOException;
import java.util.Iterator;
import java.util.StringTokenizer;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.util.StringUtils;

public class GramCount {
	
	public static class GramCountMapper extends MapReduceBase implements
			Mapper<LongWritable, Text, Text, IntWritable> {

		// count of occurrence (1 for each appearance)
		private final static IntWritable one = new IntWritable(1);
		private Text gram = new Text();

		@Override
		public void map(LongWritable key, Text value,
				OutputCollector<Text, IntWritable> output, Reporter reporter)
				throws IOException {

			String line = value.toString();
			// split string by " \t\n\r\f"
			StringTokenizer tokenizer = new StringTokenizer(line);

			if (tokenizer.countTokens() <= 2) {
				gram.set(line);
				output.collect(gram, one);
				return;
			}

			String first = tokenizer.nextToken();
			String second = "";
			
			while (tokenizer.hasMoreTokens()) {
				second = tokenizer.nextToken();
				
				gram.set(StringUtils.join(" ", new String[] { first, second })
						.trim());
				output.collect(gram, one);

				first = second;
			}

		}

	}

	public static class GramCountReduce extends MapReduceBase implements
			Reducer<Text, IntWritable, Text, IntWritable> {

		@Override
		public void reduce(Text key, Iterator<IntWritable> values,
				OutputCollector<Text, IntWritable> output, Reporter reporter)
				throws IOException {

			int sum = 0;

			while (values.hasNext()) {
				sum += values.next().get();
			}

			output.collect(key, new IntWritable(sum));
		}

	}
	
	public static void main(String[] args) throws Exception {
		JobConf conf = new JobConf(GramCount.class);
		conf.setJobName("gram_count");
		
		conf.setOutputKeyClass(Text.class);
		conf.setOutputValueClass(IntWritable.class);
		
		conf.setMapperClass(GramCountMapper.class);
		conf.setCombinerClass(GramCountReduce.class);
		conf.setReducerClass(GramCountReduce.class);
		
		conf.setInputFormat(TextInputFormat.class);
		conf.setOutputFormat(TextOutputFormat.class);
		
		System.out.println(args[0] + " " + args[1]);
		
		FileInputFormat.setInputPaths(conf, new Path(args[1]));
		FileOutputFormat.setOutputPath(conf, new Path(args[2]));
		
		JobClient.runJob(conf);
	}
}
