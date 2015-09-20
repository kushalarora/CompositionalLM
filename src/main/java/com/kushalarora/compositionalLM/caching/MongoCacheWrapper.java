package com.kushalarora.compositionalLM.caching;

import com.google.common.reflect.TypeToken;
import com.google.gson.Gson;
import com.kushalarora.compositionalLM.lang.IInsideOutsideScore;
import com.kushalarora.compositionalLM.lang.Sentence;
import com.kushalarora.compositionalLM.options.Options;
import com.mongodb.*;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import lombok.SneakyThrows;
import org.bson.Document;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import static com.mongodb.client.model.Filters.eq;
import static com.mongodb.client.model.Filters.in;

/**
 * Created by arorak on 9/19/15.
 */


 public abstract class MongoCacheWrapper<K, V> extends CacheWrapper<K, V> {

    MongoClient client;
    Gson gson;
    DBCollection coll;
    public MongoCacheWrapper(Options op) {
        super();
        client = new MongoClient();
        DB db = client.getDB("ioscoreDB");
        coll = db.getCollection("ioscoreColl");
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
        Cursor cur = coll.find(new BasicDBObject("key", getKeyString(input)));
        if (!cur.hasNext()) {
            return null;
        }
        DBObject obj = cur.next();
        byte[] bytes = (byte[])obj.get("val");
        try {
            ObjectInputStream is = new ObjectInputStream(new ByteArrayInputStream(bytes));
            return (V) is.readObject();
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    @Override
    public void close() {
        // do nothing
    }
}
