from sqlalchemy import create_engine
# from environs import Env
#
# env = Env()
# env.read_env()

DB_NAME = 'habrdb'
DB_HOST = 'localhost:5432'
DB_PASSWORD = 'pgpwd4habr'
DB_USER = 'habrpguser'
DB_ENGINE = 'postgresql+psycopg2'

ENGINE = create_engine('{0}://{1}:{2}@{3}/{4}'.format(DB_ENGINE, DB_USER,
                                                      DB_PASSWORD, DB_HOST, DB_NAME), pool_pre_ping=True)

if __name__ == "__main__":
    # Проверка на подключение к БД
    ENGINE.connect()
