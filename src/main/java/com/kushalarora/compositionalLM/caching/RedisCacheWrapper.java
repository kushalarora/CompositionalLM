package com.kushalarora.compositionalLM.caching;

import com.kushalarora.compositionalLM.options.Options;
import com.lambdaworks.redis.RedisClient;
import com.lambdaworks.redis.RedisConnection;
import com.lambdaworks.redis.codec.RedisCodec;
import org.apache.commons.io.output.ByteArrayOutputStream;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.concurrent.TimeUnit;

/**
 * Created by karora on 7/26/15.
 */
public abstract class RedisCacheWrapper<K, V> extends CacheWrapper<K, V> {

    public class SerializedObjectCodec extends RedisCodec<String, Object> {
        private Charset charset = Charset.forName("UTF-8");

        @Override
        public String decodeKey(ByteBuffer bytes) {
            return charset.decode(bytes).toString();
        }

        @Override
        public Object decodeValue(ByteBuffer bytes) {
            try {
                byte[] array = new byte[bytes.remaining()];
                bytes.get(array);
                ObjectInputStream is = new ObjectInputStream(new ByteArrayInputStream(array));
                return is.readObject();
            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }
        }

        @Override
        public byte[] encodeKey(String key) {
            return charset.encode(key).array();
        }

        @Override
        public byte[] encodeValue(Object value) {
            try {
                ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                ObjectOutputStream os = new ObjectOutputStream(bytes);
                os.writeObject(value);
                return bytes.toByteArray();
            } catch (IOException e) {
                e.printStackTrace();
                return null;
            }
        }
    }

    RedisConnection<String, Object> redisConnection;

    public RedisCacheWrapper(Options op) {
        RedisClient client = new RedisClient(op.cacheOp.cacheServer, op.cacheOp.cachePort);
        redisConnection = client.connect(new SerializedObjectCodec());
        redisConnection.setTimeout(5, TimeUnit.MINUTES);
    }


    @Override
    public void put(K input, V value) {
        redisConnection.set(getKeyString(input), value);
    }

    @Override
    public V getRoutine(K input) {
        String key = getKeyString(input);
        if (!redisConnection.exists(key)) {
            return null;
        }
        return (V) redisConnection.get(key);
    }

    @Override
    public void close() {
        redisConnection.close();
    }

    public void clear() throws Exception {
        redisConnection.flushall();
    }
}
