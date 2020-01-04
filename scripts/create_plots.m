function create_plots(fileName)
    % fileName: path della grid search contenente tutti i modelli
    d = dir(fileName);

    dfolders = d([d(:).isdir]==1);

    dfolders = dfolders(~ismember({dfolders(:).name},{'.','..'}));

    allNames = { dfolders.name };

    for directory = allNames
        fname = fileName + '/' + directory + '/data.json';
        fid = fopen(fname);
        raw = fread(fid,inf);
        str = char(raw');
        fclose(fid);

        value = jsondecode(str);

        lambda_regularization = value.learning_algorithm.lambda_regularization;
        learning_rate = value.learning_algorithm.learning_rate;
        alpha_momentum = value.learning_algorithm.alpha_momentum;
        topology = value.topology ;
        names = fieldnames(value.topology);
        plot_title= "reg:" + lambda_regularization + " lr: " + learning_rate + " mom: " + alpha_momentum;
        plot_title= plot_title+ " tpgy: (";

        for index = 1:numel(names)

            plot_title= plot_title+ topology.(names{index}).nodes + "," ;
        end
        plot_title = eraseBetween(plot_title,strlength(plot_title),strlength(plot_title));
        plot_title= plot_title+ ")";
        for i = [1,2,3,4,5]
            validation_error = readNPY(fileName + '/' + directory + "/fold-" + i +"/validation_error.npy");
            training_error = readNPY(fileName + '/' + directory + "/fold-" + i +"/training_error.npy");
            val_size = size(validation_error);
            train_size = size(training_error);
            if(val_size(2) ~= train_size(2))
                training_error(end) = [];
            end

            length = size(validation_error);
            length = length(2);
            x = 1 : length ;
            figure('Visible','off');

            plot(x,validation_error,x,training_error);
            set(gcf,'Visible','off','CreateFcn','set(gcf,''Visible'',''on'')')
            title(plot_title);
            saveas(gcf,fileName + '/' + directory + '/fold-' + i +'/learning_curve','fig');

        end
    end
