java -javaagent:lib/jamm.jar -Xmx26g -jar out/CompositionalLM-1.0-SNAPSHOT-jar-with-dependencies.jar -train -validate  -grammarType stanford -grammarFile src/resources/englishPCFG.ser.gz -saveOutputModelSerialized output/mssc_model.ser.gz -parallel -debug
