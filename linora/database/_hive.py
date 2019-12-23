import pandas as pd
import sqlalchemy as sa

__all__ = ['HiveEngine']

class HiveTableTypes():
    BigInteger = sa.types.BigInteger
    Boolean = sa.types.Boolean
    DATE = sa.types.DATE
    LargeBinary = sa.types.LargeBinary
    String = sa.types.String
    

class HiveEngine():
    def __init__(self, cur):
        self.engine = sa.create_engine(cur)
        self.tpyes = HiveTableTypes
    
    def create_table(self, table, feature, sep=','):
        t = ', '.join([i[0]+' '+i[1]+" comment '"+i[2]+"'" for i in feature])
        sql = f"CREATE TABLE IF NOT EXISTS {table} ({t}) row format delimited fields terminated by '{sep}'"
        return pd.io.sql.execute(sql, self.engine)
    
    def drop_table(self, table):
        sql = f"DROP TABLE IF EXISTS {table}"
        return pd.io.sql.execute(sql, self.engine)
    
    def alter_table_comment(self, table, comment='', property_name=''):
        sql = f"ALTER TABLE {table} SET TBLPROPERTIES('comment'='{comment}', 'property_name'='{property_name}')"
        return pd.io.sql.execute(sql, self.engine)
    
    def alter_table_delimiter(self, table, delimiter):
        sql = f"ALTER TABLE {table} SET SERDEPROPERTIES('field.delim'='{delimiter}')"
        return pd.io.sql.execute(sql, self.engine)
    
    def alter_table_name(self, old_table, new_table):
        sql = f"ALTER TABLE {old_table} RENAME TO {new_table}"
        return pd.io.sql.execute(sql, self.engine)
    
    def alter_column(self, table, old_column, new_column, dtype, comment):
        sql = f"ALTER TABLE {table} CHANGE COLUMN {old_column} {new_column} {dtype} COMMENT '{comment}'"
        return pd.io.sql.execute(sql, self.engine)
