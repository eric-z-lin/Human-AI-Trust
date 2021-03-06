# Generated by Django 3.1.2 on 2020-11-01 20:52

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ModelDataPoint',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ],
        ),
        migrations.CreateModel(
            name='ModelUserResponse',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('field_data_point_string', models.CharField(help_text='Unique string to specify the input feature combo', max_length=20)),
                ('field_ml_accuracy', models.DecimalField(decimal_places=4, help_text='ML accuracy at time of question', max_digits=4)),
                ('field_ml_confidence', models.DecimalField(decimal_places=4, help_text='ML confidence at time of question', max_digits=4)),
                ('field_ml_prediction', models.BinaryField(help_text='Actual ML prediction')),
                ('field_user_prediction', models.BinaryField(help_text='User prediction')),
                ('field_user_did_update', models.BinaryField(help_text='Whether or not user updated the model')),
                ('field_user_disagree_reason_choices', models.CharField(blank=True, choices=[('a', 'The model is typically wrong in this class'), ('b', 'The model is generally incorrect'), ('c', 'The model displayed low confidence'), ('d', 'I was confident I was right based on the current input/info'), ('e', 'Other: Free input')], default='m', help_text='If user does not use model prediction, ask why', max_length=1)),
                ('field_user_disagree_reason_freetext', models.TextField(help_text='If user chose "other", provide freetext box')),
            ],
        ),
    ]
