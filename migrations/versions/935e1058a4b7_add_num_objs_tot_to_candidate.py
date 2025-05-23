"""add num_objs_tot to candidate

Revision ID: 935e1058a4b7
Revises: f1060d175a12
Create Date: 2021-03-23 18:10:19.706399

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '935e1058a4b7'
down_revision = 'f1060d175a12'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('candidate', sa.Column('num_objs_tot', sa.Integer(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('candidate', 'num_objs_tot')
    # ### end Alembic commands ###
