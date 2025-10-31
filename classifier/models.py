from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to="uploads/")
    predicted_label = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return self.predicted_label or "Unlabeled Image"
