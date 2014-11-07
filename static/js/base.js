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

            if (!data.similar.length) {
                $('#prediction-similar').hide();
            } else {
                $('#prediction-similar').show();
                $('#prediction-similar').html('<h3>Similar series</h3><ul class="similar">' + data.similar.map(function(series) {
                    return '<li>' + series.name + '<img src="http://image.tmdb.org/t/p/w500' + series.poster_path + '"></li>';
                }).join(' ') + '</ul>');
            }
            dialogTextareaBackground.css('background-image', "url('../static/images/" + prediction + ".jpg')");
            dialogTextareaBackground.css('background-size', "100%");

        });
    });
});