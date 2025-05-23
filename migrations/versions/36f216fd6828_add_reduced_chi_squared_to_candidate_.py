"""add inside outside chi_squared to candidate_level3

Revision ID: 36f216fd6828
Revises: e25031a7b360
Create Date: 2021-04-23 12:12:47.264458

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '36f216fd6828'
down_revision = 'e25031a7b360'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('candidate_level3', sa.Column('chi_squared_inside_1tE_best', sa.Float))
    op.add_column('candidate_level3', sa.Column('chi_squared_inside_2tE_best', sa.Float))
    op.add_column('candidate_level3', sa.Column('chi_squared_inside_3tE_best', sa.Float))
    op.add_column('candidate_level3', sa.Column('chi_squared_outside_1tE_best', sa.Float))
    op.add_column('candidate_level3', sa.Column('chi_squared_outside_2tE_best', sa.Float))
    op.add_column('candidate_level3', sa.Column('chi_squared_outside_3tE_best', sa.Float))
    op.add_column('candidate_level3', sa.Column('num_epochs_inside_1tE_best', sa.Integer))
    op.add_column('candidate_level3', sa.Column('num_epochs_inside_2tE_best', sa.Integer))
    op.add_column('candidate_level3', sa.Column('num_epochs_inside_3tE_best', sa.Integer))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('candidate_level3', 'chi_squared_inside_1tE_best')
    op.drop_column('candidate_level3', 'chi_squared_inside_2tE_best')
    op.drop_column('candidate_level3', 'chi_squared_inside_3tE_best')
    op.drop_column('candidate_level3', 'chi_squared_outside_1tE_best')
    op.drop_column('candidate_level3', 'chi_squared_outside_2tE_best')
    op.drop_column('candidate_level3', 'chi_squared_outside_3tE_best')
    op.drop_column('candidate_level3', 'num_epochs_inside_1tE_best')
    op.drop_column('candidate_level3', 'num_epochs_inside_2tE_best')
    op.drop_column('candidate_level3', 'num_epochs_inside_3tE_best')
    # ### end Alembic commands ###
