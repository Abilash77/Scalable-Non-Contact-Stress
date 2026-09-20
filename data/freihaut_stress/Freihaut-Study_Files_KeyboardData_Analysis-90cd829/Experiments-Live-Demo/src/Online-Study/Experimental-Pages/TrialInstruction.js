import React, {Component} from 'react';


import ProceedButton from '../Layout-Components/ProceedButton';

export default class TrialInstruction extends Component {


    componentDidMount() {
        window.scrollTo(0, 0);
    }


    renderInstruction() {

        let phase = this.props.name.slice(0, 2);
        // Return the Correct Instruction Text
        if (phase === "Pr") {
            // if its the practice phase
            return(
                <div className="content">
                    <h3>Practice</h3><br/>
                    <h4>
                        The following is about familiarizing yourself with the tasks.
                    </h4>
                    <h4>
                        Before each task you will receive information on how to work on the respective task. Please read this information carefully.
                    </h4>
                    <br/>
                    <h4>
                        Once you are ready to start the exercise, please press „continue”.
                    </h4>
                    <ProceedButton onClick={() => this.props.proceedPhase(this.props.name)}/>
                </div>
            )
        } else if (phase === "Co") {
            // if its the condition phase render an instruction based on the condition
            if (this.props.condition === 0) {
                // if its the high-stress condition
                return (
                    <div className="content">
                        <h3>
                            Test
                        </h3>
                        <br/>
                        <h4>
                            Now, a performance test takes place.
                        </h4>
                        <h4>
                            The errors made in the test allow to assess one facet of your intelligence. Upon completion of the test, your performance will be displayed.
                        </h4>
                        <h4>
                            Another task is added the previously practiced tasks, which will be presented next. During the text, all tasks start immediately without further instructions.
                        </h4>
                        <h4>
                            Please complete all tasks as quickly and as correcly as possible.
                        </h4>
                        <ProceedButton onClick={() => this.props.proceedPhase(this.props.name)}/>
                    </div>
                )
            } else if (this.props.condition === 1) {
                // if its the low-stress condition
                return (
                    <div className="content">
                        <h3>
                            Application
                        </h3>
                        <br/>
                        <h4>
                            Now, an application takes place.
                        </h4>
                        <h4>
                            The experience gained in the application can be useful to you for processing computer-based tasks in everyday life. After completing the application, you will receive feedback.
                        </h4>
                        <h4>
                            Another task is added the previously practiced tasks, which will be presented next. During the text, all tasks start immediately without further instructions.
                        </h4>
                        <h4>
                            Please complete all tasks as quickly and as correcly as possible.
                        </h4>
                        <ProceedButton onClick={() => this.props.proceedPhase(this.props.name)}/>
                    </div>
                )
            }
        }
    }

    render() {
        return (
            <div className="section">
                {this.renderInstruction()}
            </div>
        );
    }

}