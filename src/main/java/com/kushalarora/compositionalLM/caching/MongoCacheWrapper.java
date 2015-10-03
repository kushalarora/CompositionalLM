package com.kushalarora.compositionalLM.caching;

import com.google.gson.Gson;
import com.kushalarora.compositionalLM.options.Options;
import com.mongodb.*;
import lombok.SneakyThrows;

import java.io.*;

/**
 * Created by arorak on 9/19/15.
 */


 public abstract class MongoCacheWrapper<K, V> extends CacheWrapper<K, V> {

    MongoClient client;
    Gson gson;
    DBCollection coll;
    public MongoCacheWrapper(Options op) {
        super();
        client = new MongoClient(op.cacheOp.cacheServer, op.cacheOp.cachePort);
        DB db = client.getDB(op.cacheOp.mongodbDatabase);
        coll = db.getCollection(op.cacheOp.mongodbDatabase);
        coll.createIndex(new BasicDBObject("key", 1));
    }

    @SneakyThrows
    @Override
    public void put(K input, V value) {
        BasicDBObject obj = new BasicDBObject();
        obj.put("key", getKeyString(input));
        try {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            ObjectOutputStream stream = new ObjectOutputStream(bos);
            stream.writeObject(value);
            obj.put("val", bos.toByteArray());
        } catch (Exception e) {
            e.printStackTrace();
        }
        coll.insert(obj);
    }

    @Override
    public V getRoutine(K input) {
        BasicDBObject query = new BasicDBObject("key", getKeyString(input));
        Cursor cur = coll.find(query);
        if (!cur.hasNext()) {
            return null;
        }
        DBObject obj = cur.next();
        byte[] bytes = (byte[])obj.get("val");
        try {
            ObjectInputStream is = new ObjectInputStream(new ByteArrayInputStream(bytes));
            return (V) is.readObject();
        } catch (InvalidClassException ex) {
            coll.remove(query);
            ex.printStackTrace();
            return null;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    @Override
    public void close() {
        client.close();
    }
}
