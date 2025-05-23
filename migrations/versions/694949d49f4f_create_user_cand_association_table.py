"""create user_cand association table

Revision ID: 694949d49f4f
Revises: 26570fa16077
Create Date: 2021-03-05 17:49:17.218446

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '694949d49f4f'
down_revision = '26570fa16077'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('user_candidate_association',
                    sa.Column('user_id', sa.Integer(), nullable=True),
                    sa.Column('candidate_id', sa.String(length=128), nullable=False),
                    sa.ForeignKeyConstraint(['candidate_id'], ['puzle.candidate.id'], ),
                    sa.ForeignKeyConstraint(['user_id'], ['puzle.user.id'], ),
                    schema='puzle'
                    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('user_candidate_association', schema='puzle')
    # ### end Alembic commands ###
