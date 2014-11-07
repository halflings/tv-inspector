// error notification zone
var errorMessage = function(msg) {
    error_bar.text(msg);
    error_bar.slideDown(300).delay(3000).slideUp(300);
};

$(document).ready(function() {
    error_bar = $('#error-bar');
    var dialogTextareaBackground = $('#dialog-text-wrapper');
    $('#prediction-button').click(function() {
        var dialogText = $('#dialog-text').val();
        $.post('/predict', {dialog: dialogText}, function(data) {
            var prediction = data.prediction;
            var userFriendlyPrediction = prediction.replace(/_/g, " ");
            $('#prediction-result').text(userFriendlyPrediction);
            dialogTextareaBackground.css('background-image', "url('../static/images/" + prediction + ".jpg')");
            dialogTextareaBackground.css('background-size', "100%");

        });
    });
});