java -javaagent:lib/jamm.jar -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5005 -Xmx8g -jar out/CompositionalLM-1.0-SNAPSHOT-jar-with-dependencies.jar -train src/resources/train100 -validate src/resources/valid100 -grammarType stanford -grammarFile src/resources/englishPCFG.ser.gz -saveOutputModelSerialized src/resources/ptb_model.ser.gz
