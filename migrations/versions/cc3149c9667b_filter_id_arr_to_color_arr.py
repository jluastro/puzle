"""filter id arr to color arr

Revision ID: cc3149c9667b
Revises: eadd87d54a48
Create Date: 2021-03-23 18:34:28.252841

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'cc3149c9667b'
down_revision = 'eadd87d54a48'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('candidate', 'filter_id_arr')
    op.add_column('candidate', sa.Column('color_arr', sa.ARRAY(sa.String(length=8))))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('candidate', 'color_arr')
    op.add_column('candidate', sa.Column('filter_id_arr', sa.ARRAY(sa.String(length=8))))
    # ### end Alembic commands ###
