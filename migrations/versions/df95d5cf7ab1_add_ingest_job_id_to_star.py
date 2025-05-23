"""add ingest_job_id to star

Revision ID: df95d5cf7ab1
Revises: 7df8564a73ce
Create Date: 2021-01-07 12:32:35.734399

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'df95d5cf7ab1'
down_revision = '7df8564a73ce'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('star', sa.Column('ingest_job_id', sa.BigInteger(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('star', 'ingest_job_id')
    # ### end Alembic commands ###
