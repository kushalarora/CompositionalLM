java -javaagent:lib/jamm.jar -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5005 -Xmx5g -jar out/CompositionalLM-1.0-SNAPSHOT-jar-with-dependencies.jar -train src/resources/train10 -validate src/resources/valid10 -grammarType stanford -grammarFile src/resources/englishFactored.ser.gz -saveOutputModelSerialized src/resources/model25.ser.gz -dataparallel
