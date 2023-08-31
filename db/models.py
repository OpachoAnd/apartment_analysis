from sqlalchemy import (
    Column,
    BigInteger,
)
from sqlalchemy.dialects.postgresql import BYTEA

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ML_table(Base):
    __tablename__ = 'ml_models'
    ml_table_id = Column('ml_table_id', BigInteger, primary_key=True, autoincrement=True)
    model_lama_reg = Column('model_lama_reg', BYTEA)
    model_lama_class = Column('model_lama_class', BYTEA)
    data_df = Column('data_df', BYTEA)
