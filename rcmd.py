import tensorflow as tf

class EmbModel(tf.keras.Model):
    def __init__(self, ids, rank, l2=0):
        """
        임베딩 
        Parameters: 
            ids: list
                카테고리의 수준들의 id 리스트
            rank: int
                임베딩 벡터의 사이즈
            l2: float
                l2 규제 계수
        """
        super().__init__()
        self.lu_ids = tf.keras.layers.IntegerLookup(vocabulary=tf.constant(ids))
        
        if l2 > 0:
            reg = tf.keras.regularizers.L2(l2)
        else:
            reg = None
        self.emb = tf.keras.layers.Embedding(len(ids) + 1, rank, embeddings_regularizer=reg)
    
    def call(self, x, training=False):
        x = self.lu_ids(x)
        return self.emb(x, training=training)
    
class MeanModel(tf.keras.Model):
    def __init__(self, mean, user_mean_model, movie_mean_model):
        super().__init__()
        self.mean = tf.constant([mean], dtype=tf.float32)
        self.user_mean_model = user_mean_model
        self.movie_mean_model = movie_mean_model
        
    def call(self, x, training=False):
        return self.mean + self.user_mean_model(x['userId'], training=training) + \
            self.movie_mean_model(x['movieId'], training=training)
    
# 제공한 모델 각각의 예측 결과를 더하는 모델을 만듭니다.
class AdditiveModel(tf.keras.Model):
    def __init__(self, models):
        """
        Parameters:
            models: list
                tf.keras.Model 객체로 이루진 리스트입니다.
        """
        super().__init__()
        self.models = models
        
    def call(self, x, training=False):
        # 각각의 모델에서 나온 출력을 모으기 위한 리스트 입니다.
        y_hat = []
        for i in self.models:
            y_hat.append(i(x, training=training))
        return tf.reduce_sum(y_hat, axis=0)
    
class UserHistModel(tf.keras.Model):
    """
    사용자가 이전에 평가한 영화와 평점을 입력 받는 모델입니다.
    """
    def __init__(self, user_model, movie_model, rank, l2=0):
        """
        Parameters
            user_model: tf.keras.Model
                사용자 모델
            movie_model: tf.keras.Model
                영화 모델
            rank: int
                출력 벡터의 수
            l2: float
                L2 규제, 0일 때는 규제를 사용하지 않습니다.
        """
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        
        if l2 > 0:
            reg = tf.keras.regularizers.L2(l2)
        else:
            reg = None
        
        # Rank 벡터를 만들어 내기 위한 밀집 신경망을 구성합니다. 
        # 첫번째 은닉층(1st Hidden Layer )
        self.dl = tf.keras.layers.Dense(
            64, activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(), # relu와 같은 0이상의 값만을 갖는 활성화함수는 초기화 방밥에는 HeNormal이 유용
            kernel_regularizer=reg
        )
        
        # 출력층(Output Layer)
        self.o = tf.keras.layers.Dense(
            rank, 
            kernel_regularizer=reg
        )
        
        # 사용자 벡터, 이전 시청 영화 벡터, 평점을 결합하기 위한 결합층(Concatenate Layer)을 생성합니다.
        self.cc = tf.keras.layers.Concatenate(axis=-1)

    def call(self, x, prev_movieId, prev_rating, training=False):
        vec = self.cc([
            self.user_model(x, training=training), # 사용자 벡터를 가져옵니다. N×rank
            self.movie_model(prev_movieId, training=training), # 이전 시청 영화 벡터를 가져옵니다. N×rank
            tf.expand_dims(
                prev_rating, # N
                axis=-1
            ) # 이전 평점. N×1
        ]) # N×(2×rank + 1)
        
        vec = self.dl(vec) # 첫번째 은닉층. N×64
        return self.o(vec) # 출력층. N×rank
    
class MovieInfoModel(tf.keras.Model):
    def __init__(self, df_movieinfo, movie_model, emb_config, rank, l2=0):
        super().__init__()
        self.lu_movie = tf.keras.layers.IntegerLookup(
            vocabulary=df_movieinfo.index[1:].values
        )
        self.genres = tf.ragged.constant(df_movieinfo['genres'])
        self.collection = tf.constant(df_movieinfo['collection'])
        self.ov_emb = tf.constant(df_movieinfo['ov_emb'].tolist())
        self.movie_model = movie_model
        if l2 > 0:
            reg = tf.keras.regularizers.L2(l2)
        else:
            reg = None
        
        self.emb_genre = tf.keras.layers.Embedding(
            df_movieinfo['genres'].explode().max() + 1, 
            emb_config['genre'], 
            embeddings_regularizer=reg
        )
        self.emb_collection = tf.keras.layers.Embedding(
            df_movieinfo['collection'].max() + 1, 
            emb_config['collection'], 
            embeddings_regularizer=reg
        )

        self.dl = tf.keras.layers.Dense(
            64, activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(), 
            kernel_regularizer=reg
        )
        self.o = tf.keras.layers.Dense(rank)
        self.cc = tf.keras.layers.Concatenate(axis=-1)
    
    def call(self, x, training=False):
        x_movie = self.movie_model(x, training=training)
        
        x =  self.lu_movie(x)
        x_genre = tf.gather(self.genres, x)
        x_genre = self.emb_genre(x_genre, training=training)
        x_collection = tf.gather(self.collection, x)
        x_collection =self.emb_collection(x_collection, training=training)
        x_ov_emb = tf.gather(self.ov_emb, x)
        
        x = self.cc([x_movie, tf.reduce_mean(x_genre, axis=-2), x_collection, x_ov_emb])
        x = self.dl(x)
        return self.o(x)

class UserHistModel2(tf.keras.Model):
    def __init__(self, user_model, movie_model, rank, l2 = 0, rnn = "lstm"):
        """
        Parameters
            user_model: tf.keras.Model
                사용자 모델
            movie_model: tf.keras.Model
                영화 모델
            rank: int
                출력 벡터의 수
            l2: float
                L2 규제, 0일 때는 규제를 사용하지 않습니다.
            rnn: str
                시청이력에서 사용할 RNN 종류
        """
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        if rnn == "lstm":
            self.rnn = tf.keras.layers.LSTM(32)
        elif rnn == "gru":
            self.rnn = tf.keras.layers.GRU(32)
        else:
            self.rnn = None
        if l2 > 0:
            reg = tf.keras.regularizers.L2(l2)
        else:
            reg = None
        self.dl = tf.keras.layers.Dense(
            64, activation='relu', 
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=reg
        )
        self.o = tf.keras.layers.Dense(
            rank,
            kernel_regularizer=reg
        )
        self.cc = tf.keras.layers.Concatenate(axis=-1)
        self.cc2 = tf.keras.layers.Concatenate(axis=-1)

    def call(self, x, prev_movieIds, prev_ratings, training=False):
        hist_vec = self.cc2([
            self.movie_model(prev_movieIds, training=training), 
            prev_ratings
        ])
        if self.rnn != None:
            hist_vec = self.rnn(hist_vec, training=training)
        else:
            hist_vec = tf.reduce_mean(hist_vec, axis= -2)
        vec = self.cc([
            self.user_model(x, training=training), 
            hist_vec
        ])
        vec = self.dl(vec)
        return self.o(vec)