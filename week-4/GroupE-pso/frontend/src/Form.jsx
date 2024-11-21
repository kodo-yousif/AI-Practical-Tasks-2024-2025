import { useForm } from "react-hook-form"


function Form({ params, setParams, handleInputChange, handleStartSimulation }) {
    const { register, handleSubmit } = useForm()

    const onSubmit = (data) => {
        handleStartSimulation()
    }

    return (
        <form
            onSubmit={handleSubmit(onSubmit)}
            className="flex w-[80%] justify-between flex-col gap-y-4 items-center"
        >
            <div className="flex w-[80%] justify-between ">
                <label htmlFor="">
                    Number of Particles:
                </label>
                <input
                    {...register("num_particles")}
                    className="px-4 py-2 rounded-lg text-white font-medium"
                    type="number"
                    value={params?.num_particles}
                    onChange={handleInputChange}
                />
            </div >


            <div className="flex w-[80%] justify-between">
                <label htmlFor="">
                    Goal X:
                </label>
                <input
                    {...register("goal_x")}
                    className="px-4 py-2 rounded-lg text-white font-medium"
                    type="number"
                    value={params?.goal_x}
                    onChange={handleInputChange} />
            </div >


            <div className="flex w-[80%] justify-between">
                <label htmlFor="">
                    Goal Y:
                </label>
                <input
                    {...register("goal_y")}
                    className="px-4 py-2 rounded-lg text-white font-medium"
                    type="number"
                    value={params.goal_y}
                    onChange={handleInputChange} />
            </div >


            <div className="flex w-[80%] justify-between">
                <label htmlFor="">
                    Cognitive Coefficient:
                </label>
                <input
                    {...register("cognitive_coeff")}
                    className="px-4 py-2 rounded-lg text-white font-medium"
                    type="number"
                    value={params.cognitive_coeff}
                    onChange={handleInputChange}
                />
            </div >


            <div className="flex w-[80%] justify-between">
                <label htmlFor="">
                    Social Coefficient:
                </label>
                <input
                    {...register("social_coeff")}
                    className="px-4 py-2 rounded-lg text-white font-medium"
                    type="number"
                    value={params.social_coeff}
                    onChange={handleInputChange} />
            </div >


            <div className="flex w-[80%] justify-between">
                <label htmlFor="">
                    Inertia:
                </label>
                <input
                    {...register("inertia")}
                    className="px-4 py-2 rounded-lg text-white font-medium"
                    type="number"
                    value={params.inertia}
                    onChange={handleInputChange} />
            </div >


            <div className="flex w-[80%] justify-between">
                <label htmlFor="">
                    Iterations:
                </label>
                <input
                    {...register("iterations")}
                    className="px-4 py-2 rounded-lg text-white font-medium"
                    type="number"
                    value={params.iterations}
                    onChange={handleInputChange} />
            </div >

            <input
                className="bg-blue-500 w-max py-3 px-6 rounded-lg mt-8 font-semibold text-xl"
                type="submit"
                placeholder={"Start Simulation"}
            />

        </form>
    )
}

export default Form